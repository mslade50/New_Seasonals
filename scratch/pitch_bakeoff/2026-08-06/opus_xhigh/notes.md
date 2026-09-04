# Bake-off run notes - 2026-08-06, variant opus_xhigh

Delivered: 3 ideas, all grade B, zero grade C. 19 candidates generated,
16 killed in falsification, 7 checking agents plus 3 checks I ran myself.
Dry-run HTML in `pitch.html`, payload in `ideas.json`, every script under
`checks/` with its captured output next to it.

## Novelty axes explored, and why

All seven axes got used in ideation. The ones that earned their place:

- **`inversion`** (2 candidates: XLU/SPY, USO/XLE). Chosen because the tape
  handed me two textbook "obvious mean reversion" setups (a defensive sector at
  a 21-day rank of 6 and crude at a 5-day rank of 1.6), and the fastest way to
  learn something was to ask whether the continuation side was the real one. My
  triage said yes for both. Both then died on the same vector: the apparent edge
  was one leg's beta plus overlapping windows.
- **`event_fingerprint`** (2: payroll cross-asset, pre-CPI drift). This is the
  axis that produced the day's only survivor, and it is the axis I would lean on
  first tomorrow. Monthly macro prints give naturally independent episodes, which
  is exactly what every other candidate lacked. The unconditional versions of
  both families are dead, which is the point: the payoff was in a conditional
  cell, not in the calendar.
- **`interaction_cell`** (6: dial velocity, SMH bounce, gold thrust, EEM turn,
  macro density, XLU outright, XLP). The most-used axis and the least productive.
  Every one of these was "extreme reading A while condition B" and every one had
  a small, clustered N. Lesson for tomorrow: an interaction cell needs a reason
  its episodes are independent, or it is dead before it starts.
- **`relative_value`** (4: AAPL/QQQ, SLV/GLD, GDX/GLD, XLK/XLP). Picked because
  today's dispersion was genuinely unusual (XLK +11.6% over five sessions while
  XLP fell 2.3%; AAPL -8% while QQQ +8.4%). All four died, three of them because
  the "hedge" leg turned out to carry a large net beta and the pair was a levered
  index bet wearing a spread costume.
- **`instrument_translation`** (2: DX-not-UUP for the dollar, XLE-not-USO for
  crude). Half-productive: the registry's own ruling that DX futures clear the
  cost bar where UUP does not is what made idea 1 tradeable, but the XLE version
  reproduced as a 1.11-beta energy long.
- **`flow_mechanics`** (2: SVXY term-structure carry, put/call flip). One
  produced idea 2, though only after the interaction I actually pitched was
  killed and the plain single leg was graded on its own.
- **`historical_analogue`** (1: nearest-neighbour tapes). Ran it to see whether
  today had an informative precedent. It does not, and the check was worth
  running to establish that rather than to trade it.

## What killed things

Three vectors did almost all the work, and they generalise:

1. **Episode clustering.** Eleven of nineteen. A signal-day t of 3 routinely
   became an episode t of 1.3. Two candidates arrived from checkers with headline
   numbers that were pure overlap inflation, and one (XLE) had a handed-over
   headline that did not reproduce at all (+1.46% not +3.32%).
2. **The executable entry.** A pre-market pitch enters at the next open, so the
   overnight gap after the signal is unreachable. EEM lost three quarters of its
   3-session edge that way; SLV/GLD lost 44%; XLE 10-22%. Notably GLD's thrust
   cell went the *other* way (a -28bp mean gap down after 9 of 11 thrusts), which
   is why I keep reporting both bases rather than assuming.
3. **Today's configuration sitting in the cell's weak half.** Over and over. USO
   is above its 200-day average and that subset is the flat one; XLU is in the
   rising-rate subset where the edge is 5bp; SMH's window straddles the AMAT
   print and that bucket is zero; EEM would be a day-2 entry and day-2+ entries
   run -3.1%; the macro-density cell's only lift lives in observations with the
   opposite payroll geometry to today's.

Full kill list with numbers is in `ideas.json` under `killed` (19 entries,
rendered in the email footer) and the reusable lessons are in
`registry_additions.md`.

Two candidates deserve a specific mention because they were closest to shipping
and I still cut them:

- **Long SVXY on term-structure carry** was not merely unproven, it was
  sign-negative, and the trigger was continuously ON through 22-26 January 2018.
  That is the single most useful thing I learned today.
- **Pre-CPI long SPY conditioned on today's momentum state** (N=12, treatment
  +0.825% vs a same-state control of +0.106%, permutation p ~ 0.02) was a
  legitimate grade C and I would have shipped it if I had a free slot. It lost to
  idea 2 on the repetition fingerprint (both are SPY LONG / MOO / short horizon,
  so the grammar allows only one), and idea 2 has 174 episodes against its 12.

## Least confident of the three delivered

**Idea 3, short TLT.** Its modern era does not confirm it: 18 pre-2018 episodes
average -0.878% at t = -3.88, while the 9 episodes from 2018 onward average
-0.089% at t = -0.21. Everything else about it is clean (the lift over TLT's own
drift is -0.585pp at Welch -3.10, and the six-asset sweep it won clears a
permutation null at p = 0.015), but "the effect is pre-2018" is the same sentence
that killed four other candidates today, and I only tolerated it here because the
grammar's grade B explicitly admits a single-era result and because the card says
so plainly. It is also the most correlated with idea 1 - same trigger, same three
sessions, same macro impulse - which is why both cards tell McKinley to halve
them or take the dollar leg alone.

If I had to drop one idea rather than pad to three, this is the one.

Second-least confident is idea 2: the effect is real and well-sampled, but it is
18 basis points over three days, and today adds two conditions the base cell does
not require (the 10-day P/C average still at the 71st percentile, SPY's 5-day
return at the 100th). That three-way slice is N=15 at t=1.16 - same sign, no
significance. Sized at 20bp accordingly.

Idea 1 is the one I actually believe.

## What I wanted from the state file and could not get

- **The forward macro calendar is not in `calendar.events` in a usable form for
  distance arithmetic.** The state gives `td_ahead` per event, which is what I
  needed, but every check had to re-derive session distances from
  `data/macro_events.csv` against the price index. One consequence bit me: in
  `k3b_dx_exit_alignment.py` the "fires today" line prints False, because the
  2026-08-07 payroll date does not exist in a price series that ends 2026-08-05,
  so the forward-distance lookup returns the sentinel. The historical cell is
  computed correctly and today's trigger genuinely fires (5d rank 6.0, 63d rank
  87.7, payroll two sessions after the signal close, all three confirmed from the
  state file). A `sessions_ahead` field computed on the trading calendar, or a
  small helper in `macro_calendar.py` that maps an event date to a session
  offset, would remove a real footgun.
- **No cross-asset tape ranks for the things a macro idea trades.** `pitch_tape`
  covers 217 equity-ish tickers plus a handful of macro proxies, but there is no
  EURUSD, no front-month crude, no 2-year, and DX-Y.NYB is spot rather than the
  contract. I had to check DX's own ranks by hand.
- **`data/rd2_fragility_ts.parquet` ends 2026-05-07.** The brief I wrote for the
  dial-velocity check assumed it was the long research series and it is three
  months stale; `rd2_fragility.parquet` is the only one that reaches today. Worth
  either refreshing or documenting, since CLAUDE.md points research at the `_ts`
  file.
- **No live positions and no event-sleeve state** (both in `warnings`:
  `STATUS_TOKEN` unset, `event_sleeve_state.json` missing). I could state the
  event sleeve's *schedule* overlap from the prereg doc, but not what is actually
  open. For a product whose `overlap` field is a hard requirement, that is the
  gap that matters most.
- **`book.staged_signals` is empty**, so I could not check whether the scanner
  was about to stage XLE, XLU or SPY-family signals this morning and had to
  describe the overlap structurally instead.
- Minor: `history.scoreboard` and `recent_fingerprints` are empty (lifetime
  pitched = 0), so the repetition machinery was untested on this run.

## Publisher observation

The email card does not print the futures `Contract` string. The value is carried
into the Pitch tab (`TAB_COLUMNS` includes `Contract`) and idea 1's card says
"Manual only. futures leg", but the spec's "the card prints the contract for
manual entry" is not satisfied by the HTML. Flagged rather than fixed, since this
is a bake-off run and `daily_pitch.py` is out of scope.

## Bake-off compliance

- State read as-is; `scripts/build_pitch_state.py` not re-run.
- All price loads truncated at bars strictly before 2026-08-06 via
  `checks/_common.py::load`, so today's realized session never entered a check.
  (Only 2 of 1101 tickers had a 2026-08-06 bar in the cache anyway.)
- Ideas written to `scratch/pitch_bakeoff/2026-08-06/opus_xhigh/ideas.json`;
  checks under `.../checks/`; every `evidence.script` points at the file used.
- Published only with the mandated dry-run command. No email, no Sheets write,
  `data/pitch_journal.jsonl` untouched (still 0 bytes, timestamped 17:00, before
  this run), and the variant journal was correctly not created because `--dry-run`
  short-circuits the append.
- `data/pitch_negative_registry.md` untouched; additions listed in
  `registry_additions.md`.
- No files read under any other `scratch/pitch_bakeoff/` variant folder.
- Nothing committed.
