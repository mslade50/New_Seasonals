---
name: daily-pitch
description: Produce the Daily Pitch - exactly three novel, falsified trade ideas for today, delivered as an email and a Sheets approval tab. Use when running the morning pitch (scheduled ~7:00 AM ET or on request), or when McKinley asks for pitch ideas, a trade suggestion run, or a rerun of today's pitch.
---

# Daily Pitch

Deliver **exactly three** trade ideas, invented from this repo's knowledge and
interrogated against its data before they reach McKinley. He reads them, says
yes or no per idea, and places orders. There is no conversation: the ideas must
be finished when they arrive.

Spec of record: `daily_pitch_agent_spec_2026-08-06.html` (repo root). Read it
if anything here is ambiguous. Sections 4, 5, 7 and 8 are hard requirements.

## What this is not

- **Not a systematic engine.** The strategy book already is that. Do not build
  or propose an activation layer, a card library, or anything that fires on a
  rule. Every idea here is a one-off judgement call.
- **Not a re-run of the book.** An idea that is materially a `STRATEGY_BOOK`
  trade, an event-sleeve trade, or a seasonal-sheet ticket is dead on arrival
  unless it adds a real twist (different instrument, different leg structure,
  opposite side, meaningfully different window), and the write-up must name the
  overlap. Repeating the scanner has zero value.
- **Not a place orders happen.** Nothing is ever placed without McKinley typing
  Y in the Pitch tab, and the runner that reads those cells is a separate,
  activation-gated program.

False positives are fine. He filters. Unchecked ideas are not fine.

---

## Stage A. State (deterministic, already code)

```bash
python scripts/build_pitch_state.py
```

Writes `data/pitch_state.json` (calendar, risk dials, book state, earnings,
seasonality, research index, your own recent pitch history) and
`data/pitch_tape.json` (per-ticker return ranks, z10, ATR, 52w and 200d
distances for ~215 names). Read the state file whole.

Read the tape file whole as well, and SORT it. Do not look up the handful of
tickers you already have a thought about; that turns a 217-name cross-asset
picture into whatever you walked in with. Everything McKinley trades is in
there, rates and metals and energy and FX included.

**Check `warnings` first.** A stale price cache or a missing dial changes what
you can honestly claim. If the freshest bar is not the prior session, say so in
every affected idea's evidence and consider standing down to lower grades.

`history.recent_fingerprints` is what you pitched inside the last 10 trading
days. An idea matching one of those needs a `changed_since` sentence saying
what materially changed, or it must be dropped.

## The lab and the data map

### pitch_lab is the substrate. Do not rebuild it.

Every check script starts from `pitch_lab.py` at the repo root:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
```

It holds the whole falsification toolkit, tested by `tests/test_pitch_lab.py`:
price loading (`load_prices`, `close_panel`), event alignment (`load_events`,
`event_in_window`), the lag-aware forward returns (`fwd_lag`, `vehicle_ret`;
lag=1 is the real MOC-tomorrow order), declustering and controls
(`declusters`, `local_control`), `summarize` / `era_split` /
`bootstrap_p_le0` / `sign_test` / `cluster_note`, the whole round-1 kill
battery in one call (`battery`), and the development helpers (`horizon_scan`,
`episode_paths`). Conventions are in its docstring: returns are FRACTIONS in
and PERCENT out of `summarize`, ATR is Wilder-14, controls are own-drift,
all-days, and local +/-126td.

Do not re-derive any of this in the day folder. On 2026-08-07 the morning
rebuilt all of it ad hoc (`_common.py` / `_engine.py` / `_study.py`) and still
had to leave itself a warning comment about a units double-scaling bug; those
helpers were promoted into `pitch_lab` afterwards so kill statistics stay
identical across mornings. Day-local helpers are for genuinely novel machinery
only, and anything reusable gets promoted into `pitch_lab` (with a test) the
same morning.

### What a check may reach

The state and tape files are a summary, never the boundary. The repo holds:

| data | access | span | notes |
|---|---|---|---|
| daily OHLCV, ~2000 names | `pitch_lab.load_prices` (`data/master_prices.parquet`) | 1999+ | adjusted basis; SOXS is broken upstream before 2026-05-26 |
| intraday 15-min bars | `intraday_data.get_intraday(ticker)` | 2003+ | ~197 liquid names, ET stamps; splits an event reaction into overnight vs intraday, or shows where in the session an edge accrues; no caret tickers |
| macro event calendar | `pitch_lab.load_events` (`data/macro_events.csv`) | 2000-2027 | nfp, cpi, ppi, fomc_decision, fomc_minutes, opex, quad_witching, vix_expiry, jackson_hole, election |
| earnings calendar | `data/earnings_calendar.parquet` | 946 tickers | includes forward dates |
| CBOE put/call ratios | `data/cboe_putcall.parquet` | 2006-11+ | total / index / equity / etp / spx / oex; conventions in `pc_fear.py` |
| 30d implied vol by ticker | `data/iv_history.parquet` | 2024-07+ | two years: today's state, not testable history |
| vol term structure + chain | `data/option_surface_history.parquet`, `data/option_positioning_history.parquet` | accruing since 2026-08-05 | positioning context for flow ideas; useless for backtests so far |
| French factors, monthly | `data/factor_returns_monthly.parquet` | 1926+ | MktRF SMB HML RMW CMA Mom RF |
| the book's own ledger | `data/backtest_trades_full.parquet` | 23y | overlap checks and "what would the scanner do here"; overflow-tier stats carry a survivorship caveat |
| fragility dial history | `data/rd2_fragility.parquet` | 2016+ | point-in-time append-only since 2026-07-02; earlier rows are a recompute vintage, state which you used |
| sector map | `data/sector_map.parquet` | ~1460 names | |

Honesty rule for `flow_mechanics`: the only positioning data in the repo is
the accruing option_* files plus the expiry calendar. A flow idea whose
falsification needs data the repo lacks (dealer gamma history, futures roll
positioning) must say so in the check and grade accordingly. A mechanism that
cannot be measured has not been verified.

## Stage B. Survey the whole surface, then select from it

This stage failed on 2026-08-07 and the failure is worth stating, because the
old wording looked fine and was not. It said "generate 8 to 12 candidates
drawing on at least four axes". That is a menu, and a menu is satisfied by
whatever comes to mind first. The `event_fingerprint` axis got its tick from
one SPY idea, the floor passed, and nothing ever asked which assets had been
looked at. It was an August NFP in a midterm year, the spec names the
midterm-August-NFP cross-asset table as its own example of that very axis, and
every calendar-anchored check that morning ran on SPY. Nothing was missing
from the data. TLT, GLD, UUP, DX-Y.NYB and ^TNX were all sitting in the tape.

So the stage is now two halves and the first one is not optional. Map the
surface, then pick from the map. Never generate from recall.

### B1. Map the surface before generating a single candidate

Write `scratch/pitch_checks/<YYYY-MM-DD>/00_surface_map.md` FIRST. It
enumerates today's whole opportunity surface and gives every cell a verdict.
The dismissals matter as much as the picks: a cell you do not examine has to
be visibly dismissed with a reason, never silently absent. If the map is
missing, the morning is not surveyed and nothing may publish. That is enforced
on disk since 2026-08-08, on the ideas path as well as the stand-down path.

Three enumerations, each exhaustive.

**1. Every live calendar event, crossed with every asset class.** Take every
entry in `calendar.events`, the whole window and not just the next one, and
cross it with all of these:

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

Six events by ten classes is sixty cells. You are not checking sixty things.
You are deciding in writing which of the sixty deserve a check and why the
rest do not. "NFP x rates: not examined" is a legal line only when the next
words say why.

**2. Every tape extreme, by class.** Sort the whole tape. What sits at a 52w
edge, what is rank-extreme at 5/21/63d, what is furthest from its 200d, what
carries the widest z10. Name the outliers in every class, not only the classes
you already had an idea about.

**3. Every live seasonal and cycle cell**, from `seasonality` plus the
cycle-year state. Midterm is a conditioner on everything above rather than an
idea of its own.

**4. Every active watchlist entry.** `watchlist` in the state carries
near-misses parked by earlier mornings, each with the number it turned on.
Every active entry gets its own verdict line: trigger moved, CHECK, citing
today's value; trigger unchanged, PASS, still citing today's value so the
map shows it was looked at; listed under `expired`, prune it when the file is
rewritten after publish. A watch that fires is the cheapest deep dive of the
morning, because its check script already exists and only needs re-running on
today's tape.

### B2. Select 8 to 12 candidates from the map

Name the novelty axis on each; the journal tracks which axes actually earn.

| axis | what it means |
|---|---|
| `instrument_translation` | a known effect expressed through a better vehicle (SVXY instead of short UVXY, DX futures instead of UUP) |
| `interaction_cell` | event x seasonal x cycle-year x tape-state combinations no study has opened. The repo's studies are the periodic table; the compounds are mostly unexplored |
| `relative_value` | long X against short Y where the research only ever examined the legs alone |
| `inversion` | conditions under which a documented edge flips sign (September inverting the post-opex vol crush, midterm years inverting the pre-FOMC drift) |
| `historical_analogue` | nearest-neighbour tapes to today by return, vol and breadth, and what worked in the following week, with honest N |
| `flow_mechanics` | dealer gamma around expiries, futures roll, month-end rebalancing, index adds, buyback windows as the thesis engine |
| `event_fingerprint` | cross-asset behaviour around a specific calendar event applied to whatever the horizon actually serves up |

Default horizon is 1 to 10 trading days. Instruments are anything McKinley can
trade: US equities and ETFs, and futures (DX, ES family, treasuries, metals).
Never reject an idea for lacking an ETF wrapper; futures are fine and the card
prints the contract for manual entry. No options in v1.

**Coverage requirements. These are requirements, not targets.**

- Candidates must touch **at least four distinct asset classes** from the B1
  table. A morning of five equity ideas is a failed survey no matter how good
  the five ideas are.
- **Every event inside its window appears in the map with a verdict.** The
  event you skip is the one McKinley asks about, and he will be right.
- **At least one candidate anchored on a calendar event, and at least one on a
  price state.** These are different search modes. On 2026-08-07 every
  calendar-anchored check was SPY and every cross-asset check was anchored on
  a chart level (bond floor, silver catch-up, GDX drawdown, natgas at a 52w
  low). Neither mode was missing. They were never crossed, which is where the
  NFP-times-rates cell lived.
- Four axes remains the floor, but axis variety no longer substitutes for
  coverage. Both are checked.

Coverage is about where you LOOK, not about what you ship. Standing down after
a genuinely comprehensive survey is a fine morning. Shipping three good equity
ideas without having opened rates, metals or FX is not, and it is invisible in
the output unless the map says so.

Before spending a check on a candidate, run it against
`data/pitch_negative_registry.md` (also inlined in the state file under
`research.negative_registry`). A collision does not automatically kill the
idea, but the write-up must explain what is different.

The axis table is also a feedback loop, not just a menu: once `scoreboard`
in the state carries graded ideas, read the per-axis and per-grade splits
before selecting. An axis that has bled across a meaningful number of graded
ideas needs a better-than-usual reason to get another slot; one that earns
deserves extra looks. Note the read in the surface map. While the graded
count is still a handful, say that instead and move on.

## Stage C. Falsification (adversarial, the point of the whole thing)

Hand the surviving candidates to checking agents whose brief is to **kill**
them. Two to three agents in parallel, three or four candidates each, plus a
final red-team pass over the survivors together. Each check is a fast empirical
interrogation, not a backtest, and it writes a script under
`scratch/pitch_checks/<YYYY-MM-DD>/`.

### The checker brief (what each agent gets, and returns)

Checker quality must not depend on how the handoff happens to be phrased, so
every checking agent's prompt contains all of this:

- the candidate block from the surface map, verbatim, with its novelty axis
  and the cell it came from
- the path to `00_surface_map.md` and to the day's checks directory
- the negative-registry entries adjacent to the idea (not the whole file)
- the `pitch_lab` import boilerplate from "The lab" above, plus the one-line
  conventions reminder: entry is lag=1 MOC-tomorrow, fractions in / percent
  out, `battery()` is round 1
- the standing brief: your job is to kill this; a survivor is a failure to
  kill, not a success to celebrate

Each checker returns, per candidate: a verdict (KILL / SURVIVES /
NEAR-MISS), the two or three decisive numbers, the script paths it wrote,
and, for every NEAR-MISS, **the number it turned on**, because that line
feeds the watchlist. A kill also names WHICH substantive kill it is (the
list below); "weak" is not a verdict.

### Round 1. Every check answers all of these

`pitch_lab.battery()` covers 1, 2, 5 and 6 in one call; the checker's work
is choosing the mask, the legs and the controls honestly, then interpreting.

1. Does the claimed pattern exist in the data at all? Measure the window return
   around the anchor against a real control: the all-events baseline and the
   instrument's own unconditional drift over the same horizon.
2. What is N, what is the worst window, and is it era-stable (did it die after
   2018)?
3. Does it collide with the negative-results registry?
4. Does the book already hold correlated exposure, and does the idea survive
   that overlap once disclosed?
5. Cost sanity: spread, carry and roll of the vehicle against the size of the
   edge. Six basis points of edge cannot pay an ETF's drag.
6. Tomorrow-specific tail risk: is a known volatility event inside the window?

### Round 2 is mandatory for anything that survives round 1

No candidate reaches compose on one green script. The 2026-08-07 kills that
mattered all came from second-round probes: declustering flipped GDX from
+4.41% to -2.80%, a gate-matched control killed the SVXY carry, and the
b-suffix scripts did the work the headline scripts could not. So round 2 is
part of the definition of checked, not extra credit. For each round-1
survivor, probe (same script with a `b`/`c` suffix, or a section in the
original):

1. **Decluster + concentration**: episode-level stats and
   `cluster_note`: how much of the total sits in the top 2 windows or one
   year?
2. **Definition neighbours**: nudge the trigger (threshold, lookback,
   ranking window) to the nearest reasonable alternatives. If the result
   only exists under one definition, the definition is the finding.
3. **Era and regime split**: pre/post 2018 at minimum, plus whatever regime
   the mechanism implies (cycle year, rate regime, vol regime).
4. **Gate attribution**: run the idea WITHOUT its gate. If the gate does not
   move the result, nothing may be attributed to the gate, and the write-up
   must say what the trade actually keys on.

### Round 3. Develop the survivors before composing

A survivor is a pattern; a pitch is a trade. Every candidate that will be
composed gets one development script (`_dev` suffix) answering four
questions, and the composed idea must match its answers rather than the form
the candidate was first imagined in.

**Its path goes in `evidence.dev_script` and the publisher requires it** for
every composed idea, so round 3 is structural rather than something the prose
can claim. One file may serve as both `script` and `dev_script` when it
carries the development section itself. A directed idea may omit it, since
McKinley asked for the trade in a stated form and the shaping work is done.

1. **Horizon**: `pitch_lab.horizon_scan` across 1 to 10 td. `horizon_td` and
   `time_td` come FROM this table. If the edge lives at h=3 and decays by
   h=5, pitching h=5 ships a worse trade for a rounder number.
2. **Entry form**: MOC against a close-anchored LIMIT at a sensible k ATR,
   compared as WHOLE variants (fill rate plus conditional stats; never a
   marginal-fill decomposition, registry rule). Choose knowing the placement
   consequence: a close-anchored LIMIT auto-places with its bracket, MOO and
   MOC place with a time exit only.
3. **Exits**: if a target or stop is attached, show the sensitivity around
   the chosen level; if none, one line on why time-only fits the mechanism.
4. **Loser paths**: `pitch_lab.episode_paths` on the losing episodes.
   `what_kills_it` quotes a number from this ("losers averaged -0.8% by day
   2 and never bounced; a close below X by then says the thesis is wrong"),
   never a generic risk.

### The red-team pass (survivors together, after development)

One final agent over the whole surviving set, before compose:

- **Basket correlation**: are the three (or five) survivors secretly one
  macro bet? Compute the correlation of their leg returns over the hold
  window; if two ideas are one trade, say so and either merge or drop one.
- **Book overlap**: against the SYSTEMATIC layers only, which is all the
  state carries: staged scanner signals, sleeve state and the ledger. Live
  broker positions are deliberately absent from the pitch state and stay
  that way; McKinley applies his own holdings when he reads the ideas.
- **Cost recheck** at the developed entry form, not the round-1 assumption.
- For each survivor, write the strongest single argument against it. If that
  argument would convince you, it is a kill, not a footnote.

**Hard requirement.** Every delivered idea carries falsification evidence
computed fresh this morning and **at least one named consideration that could
have killed it and did not** (the `survived` field). An idea with no such line
is not finished.

### Small N is not a kill. Read this before reaching for a correction.

Markets produce small samples by construction. A monthly event has twelve
observations a year, a cycle-year cell has one every four, and by the time any
of them reaches N=50 the regime that produced it is usually gone. A rule that
demands large N does not find safer trades, it finds older ones.

The doctrine, and it is a rule rather than a mood:

- **A clean or near-clean record with a large per-event edge and a mechanism
  is evidence in itself.** 6-0 is p=0.016 under a fair coin. Quote the record
  and its exact `pitch_lab.sign_test` p alongside the per-event edge, and
  ship it at the grade its N earns. If the stated edge is large and the
  reason it shows up makes logical sense, that IS the case for the trade.
- **No idea needs a t-stat to ship.** At N<15 a t-stat is mostly noise
  anyway; its absence, or its being under 2, is never a kill and never a
  demotion. The publisher does not require `t_stat`; leave it null and let
  the sign test and the edge carry the evidence line.
- **"Insufficient N", "not statistically significant" and "t below 2" are
  illegal kill reasons standing alone.** Every small-N kill must name one of
  the substantive kills below and quote its number. A checker that kills on
  sample size alone has not done the check.
- **An idea with a plausible mechanism and N<15 is a grade C, which is
  exactly what grade C is for.** Ship it, grade it honestly, size it at the
  default and let the scoreboard settle it. Do not kill it for being small.

**Multiplicity corrections price the cost of a SEARCH, so they only apply to
cells the search found.** If you built a grid and reported its best occupant,
correct for the grid. If the idea arrived with a mechanism attached, from
McKinley, from the research docs, or from a stated prior, it was
pre-specified and there is no search to charge it for. Applying a family-wise
correction to somebody else's pre-registered hypothesis is a category error,
and it happened on 2026-08-07: a DX cell McKinley asked about by name was
scored against a 47-cell grid the checker had built itself, and killed at a
family-wise p of 0.904 when the pre-specified p was 0.011.

What still kills a small-N idea, because these are about the idea rather than
about the sample size:

- **no mechanism.** "This cell is green" with no story about who is on the
  other side is a data artifact whatever its N.
- **a filter that does not filter.** If a gate removes one observation from
  six, nothing may be attributed to that gate. Say what the trade actually
  keys on.
- **definition fragility.** If the result exists under one definition of the
  trigger and vanishes under a reasonable neighbour, the definition is the
  finding.
- **sign instability across eras**, or a mechanism that is falsified inside
  its own window.
- **cost.** A small edge is still an edge; an edge under its own round trip
  is not.

Report N plainly and let the grade carry it. "False positives are fine, he
filters" is the product, and a filter with nothing to filter is not one.

Ideas killed here are journaled with their reasons and appear as one line each
in the email footer. They are not shown as ideas. If a kill teaches something
reusable, append it to `data/pitch_negative_registry.md` the same morning.

Do not quietly shrink this stage to save tokens. Fewer, better-checked ideas
beat unchecked ones, and the delivery is still always three.

### When nothing survives

Some mornings everything dies. That is a real result and it ships as a
stand-down: an email titled NO TRADES that leads with the near-misses, an
empty Pitch tab, and a `stand_down` record in the journal. Never write false
`survived` lines to reach three, and never publish nothing at all, which is
indistinguishable from a broken scheduled task.

A stand-down costs MORE work than shipping, on purpose. The publisher enforces
all of it: at least 8 candidates over at least 4 distinct novelty axes, at
least 4 distinct asset classes, at least 6 named kills with reasons, a reason
of at least 120 characters, a `checks_dir` holding both real `.py` checks and
the `00_surface_map.md` from stage B1, and 1 to 3 near-misses each carrying
the **number it turned on**. If you are standing down because the sweep was
thin, the answer is to go back to stage B, not to fill in this block.

"Nothing worth trading today" is a claim about the whole surface, so it is the
one verdict that has to prove it surveyed the whole surface. List the classes
you covered in `asset_classes`.

Near-misses do not evaporate with the morning: every `closest` entry also
goes onto the watchlist with the number it turned on (see "After
publishing"), so the run that finds the trigger has moved starts from a
finished check instead of a blank page.

```json
{
  "asof": "YYYY-MM-DD",
  "ideas": [],
  "stand_down": {
    "reason": "what the morning looked like and why none of it survived",
    "candidates_considered": 24,
    "axes": ["relative_value", "inversion", "event_fingerprint", "flow_mechanics"],
    "asset_classes": ["us_large", "rates", "gold_miners", "energy", "dollar_fx"],
    "checks_dir": "scratch/pitch_checks/YYYY-MM-DD",
    "closest": [
      {"title": "Short TLT at the 52w low",
       "decisive": "excess +1.263% over TLT's own 2018+ downtrend, Welch t=2.10",
       "why_died": "8 of 12 episodes are 2021-22; ex-2022 t=0.69, and today's +31bp yield thrust is the 3.8th percentile of the winning episodes",
       "script": "scratch/pitch_checks/YYYY-MM-DD/r2_tlt_short_inversion.py"}
    ]
  },
  "killed": [{"title": "...", "reason": "...", "novelty_axis": "..."}]
}
```

Publish it the same way; `--validate-only` checks the floors first.

## Stage D. Compose

Pick the best three by risk and reward over their stated horizons. Grade them
honestly:

| grade | meaning |
|---|---|
| A | N >= 50, abs(t) >= 2.5, holds across eras, verified fresh today |
| B | real pattern, N 15 to 50 or single era, verified fresh today |
| C | context, not statistics: N < 15 fingerprint or analogue reasoning. **At most one per day** |

A grade-C evidence line quotes the record and its exact sign-test p ("6-0,
sign p 0.016") plus the per-event edge; `t_stat` may be null. A strong small
sample competes for the C slot on edge size and mechanism quality, and it is
allowed to win the morning.

Write `data/pitch_ideas.json` in the schema below, then publish. The publisher
enforces the grammar, so a schema mistake fails loudly instead of shipping.

### Prose rules

House style, and McKinley reads every word:

- No em dashes. No "it's not X, it's Y". No AI throat-clearing, no hedging
  filler, no restating the question.
- The thesis is three to five sentences and must contain a **variant
  perception** (what the market prices against what the evidence says) and
  **who is on the other side** (forced selling, informed disagreement, or
  neglect). If you cannot write those, the idea is not ready.
- Evidence is numbers with their N, control and era note. One table at most.
- `what_kills_it` is the observation during the week that would invalidate the
  thesis, not a generic risk disclaimer.

### Schema

```json
{
  "asof": "YYYY-MM-DD",
  "ideas": [
    {
      "title": "one line, no ticker soup",
      "grade": "A|B|C",
      "novelty_axis": "relative_value",
      "horizon_td": 5,
      "legs": [
        {"ticker": "GLD", "side": "LONG", "weight": 1.0},
        {"ticker": "SLV", "side": "SHORT", "weight": 1.0}
      ],
      "entry": {"type": "MOC"},
      "exit": {"time_td": 5, "time_order": "MOC",
               "target_atr": null, "stop_atr": null,
               "event_anchor": "optional prose, e.g. exit at the close before the decision"},
      "sizing": {"mode": "risk_bps", "risk_bps": 30, "stop_atr_for_sizing": 1.0},
      "thesis": "...",
      "evidence": {"summary": "...", "n": 22, "t_stat": 2.1,
                   "window": "2004-2025", "control": "...", "era_note": "...",
                   "table": [["cohort","N","avg","hit"], ["...","...","...","..."]],
                   "script": "scratch/pitch_checks/2026-08-06/gld_slv_nfp.py",
                   "dev_script": "scratch/pitch_checks/2026-08-06/gld_slv_nfp_dev.py"},
      "survived": "the consideration that could have killed it and did not",
      "what_kills_it": "...",
      "overlap": "what the book or sleeves already hold that correlates, or None",
      "changed_since": "only when re-pitching inside 10 td"
    }
  ],
  "killed": [{"title": "...", "reason": "...", "novelty_axis": "..."}]
}
```

Vocabularies, and nothing else is legal:

```
entry.type   MOO | MOC | LIMIT
             LIMIT also needs anchor (OPEN|CLOSE), atr_mult (signed),
             fill_window_td (1..10)
exit         time_td is ALWAYS present (1..horizon_td), time_order MOC|MOO,
             target_atr and stop_atr optional
sizing.mode  risk_bps (default 30 bps of NAV) | nav_pct (index or carry
             constructions that are not naturally risk-sized; say why in the
             thesis)
legs         side LONG|SHORT; a futures leg adds sec_type "FUT", contract
             (exact string for manual entry), proxy_ticker (a master_prices
             series for levels and grading) and multiplier (point value)
```

ATR is Wilder-14 on the traded instrument. This is deliberately not the
systematic book's simple 14-day ATR: a pitch level is never a scanner level.

Auto-placement follows from the grammar, so choose entries knowing the
consequence:

- `LIMIT` anchored to `CLOSE` is fully placeable with its stop and target,
  because the fill price is known up front.
- `MOO` and `MOC` are placeable with a time exit only. Adding a price stop or
  target to them makes the idea manual.
- `LIMIT` anchored to `OPEN` is placed in the runner's post-open pass.
- Any futures leg is manual in v1 and the card prints the contract.

## Publish

```bash
python daily_pitch.py --ideas data/pitch_ideas.json --validate-only   # check
python daily_pitch.py --ideas data/pitch_ideas.json                   # ship
```

The publisher validates, sizes every leg, captures yesterday's Approve cells
off the Pitch tab before overwriting it, sends the email, rewrites the tab, and
appends to `data/pitch_journal.jsonl`. Iterate on `--validate-only` until it is
silent. `--dry-run --html-out preview.html` renders without sending.

Fix validation errors by fixing the idea, never by loosening the grammar.

### The publish gates read the disk, not the payload

Since 2026-08-08 an ideas publish has to prove the morning happened. Three
things are checked against `scratch/pitch_checks/<asof>/` and `--validate-only`
enforces every one of them exactly like a real publish:

- the day folder exists and holds `00_surface_map.md` plus at least one `.py`
  check. No surface map means stage B1 was skipped, the morning is not
  surveyed, and nothing publishes.
- every `evidence.script` and `evidence.dev_script` resolves to a file that
  exists INSIDE that folder. A path into yesterday's folder fails as stale,
  which is the machine-checkable half of "computed fresh this morning".
- every composed idea carries `dev_script` (round 3 above).

A directed-only publish skips the first check and none of the others: the
survey rule constrains the agent, not McKinley, but a directed idea still
needs its own check written today.

The kill reasons are linted too. A reason that reads as sample size and
nothing else ("insufficient N", "not significant", "t below 2") prints a
`KILL-LINT:` line and is tagged in the email footer, per the doctrine above.
It is a warning and never blocks a publish, so treat it as a prompt to name
the substantive kill you actually found rather than an error to route around.

## After publishing

1. Append any reusable kill to `data/pitch_negative_registry.md`.
2. Update `data/pitch_watchlist.json`: append today's near-misses (NEAR-MISS
   verdicts from stage C and any stand-down `closest` entries) with title,
   cell, **the trigger number**, script path, source and an expiry (default
   15 td out; a calendar-dated trigger may park to its date, like the TLT
   floor cell parked to the first non-midterm NFP). Remove entries that
   expired or whose trigger fired today. The state builder folds this file
   into tomorrow's state, and stage B1 owes every active entry a verdict.
3. Leave the check scripts in `scratch/pitch_checks/<date>/`; the evidence
   field points at them and they are the audit trail. Promote any genuinely
   reusable helper into `pitch_lab.py` with a test.
4. Grading of prior days runs on its own:

```bash
python scripts/grade_pitch_journal.py
```

It replays every pitched idea, approved or declined, against its own stated
entry and exit, and rewrites `data/pitch_scoreboard.json` for tomorrow's email
footer. If the creative layer is not earning its keep, that is where it shows,
and the product gets killed or retuned like any other strategy.

## Standing down

If the price cache is broken, the pitch does not ship on stale data. Say so and
stop. A missed morning delivers nothing; it never delivers stale ideas late in
the session.
