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
missing, the morning is not surveyed and nothing may publish.

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

## Stage C. Falsification (adversarial, the point of the whole thing)

Hand the surviving candidates to checking agents whose brief is to **kill**
them. Two to three agents in parallel, three or four candidates each, plus a
final red-team pass over the survivors together. Each check is a fast empirical
interrogation, not a backtest, and it writes a throwaway script under
`scratch/pitch_checks/<YYYY-MM-DD>/`.

Every check answers all of these:

1. Does the claimed pattern exist in the data at all? Measure the window return
   around the anchor against a real control: the all-events baseline and the
   instrument's own unconditional drift over the same horizon. The sweep
   scripts in `scratch/` are the pattern to imitate.
2. What is N, what is the worst window, and is it era-stable (did it die after
   2018)?
3. Does it collide with the negative-results registry?
4. Does the book already hold correlated exposure, and does the idea survive
   that overlap once disclosed?
5. Cost sanity: spread, carry and roll of the vehicle against the size of the
   edge. Six basis points of edge cannot pay an ETF's drag.
6. Tomorrow-specific tail risk: is a known volatility event inside the window?

**Hard requirement.** Every delivered idea carries falsification evidence
computed fresh this morning and **at least one named consideration that could
have killed it and did not** (the `survived` field). An idea with no such line
is not finished.

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
                   "script": "scratch/pitch_checks/2026-08-06/gld_slv_nfp.py"},
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

## After publishing

1. Append any reusable kill to `data/pitch_negative_registry.md`.
2. Leave the check scripts in `scratch/pitch_checks/<date>/`; the evidence
   field points at them and they are the audit trail.
3. Grading of prior days runs on its own:

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
