# fable_high bake-off notes — 2026-08-06

Run after the close, written as a normal pre-market pitch for the 2026-08-06
session (freshest bar 2026-08-05; every check truncates at that date because
master_prices already carried a provisional 08-06 bar).

## Novelty axes explored and why

Six of the seven axes got at least one candidate:

- **event_fingerprint** — NFP is tomorrow (1 td), the dominant calendar fact
  in the window, and the repo had a fresh but unexploited study
  (scratch/august_nfp_cross_asset.py) whose one real cell (SPY Aug-NFP day0
  -36 bps) was sitting unpitched. Also the killed TLT-NFP-week idea.
- **interaction_cell** — tried to sharpen the NFP cell with the 100th-pctile
  5d run into the print (hot-tape x NFP, and hot x Aug). Both empty (N=2 or
  noise); the interaction died, the base cell survived.
- **relative_value** — the tape served two extreme relative dislocations:
  XLU 21d rank 6 vs SPY at highs, and AAPL 5d rank 1.2 vs QQQ rank 100.
- **inversion** — the XLU check inverted its own hypothesis: the dip-buy died
  and the continuation short was the evidenced side. Delivered as idea 2.
- **historical_analogue** — USO -11% week, GDX +7% day, FXI 21d rank 97,
  SPY +5.5% thrust to a high. All four killed on N or era.
- **instrument_translation** — the dollar washout expressed in DX futures
  (the registry's own example: UUP fails cost, DX passes). Delivered as
  idea 3.

Not explored: **flow_mechanics**. The honest reason: the state file carries
nothing to check dealer gamma, roll dates or buyback-window timing against,
and opex/VIX-expiry (the checkable flow anchors) are 9-11 td out, beyond the
product's default horizon. A buyback-blackout-ending thesis crossed my mind
for August and had no data surface to falsify against, so it was never a
candidate.

## Candidates killed and what killed them

Ten raw candidates, seven killed (full list with numbers in ideas.json
`killed` and registry_additions.md):

1. XLU dip-buy — wrong-signed in every cohort; became idea 2 inverted.
2. AAPL/QQQ snapback — tradeable cell indistinguishable from baseline spread
   drift (+28 vs +37 bps); extreme cell N=4.
3. TLT Aug-NFP week — legacy August bond seasonality in disguise; the
   seasonal flipped negative 2020+. This was the best-looking headline stat
   of the morning (hit 0.75) and the control fully dissolved it.
4. USO crash bounce — only positive sub-cell is N=11 with four 2026
   episodes; self-referential mining.
5. GDX spike follow-through — exact today-shape cell N=7; noise both ways.
6. FXI momentum — +2 bps 2018+. Dead.
7. SPY thrust-at-high continuation — N=5 episodes ever; context only.

The NFP hot-tape interaction also died as a sharpener (kept inside idea 1's
evidence rather than the killed list, since the base cell shipped).

## Least confident of the three delivered

**Idea 3, the DX washout long.** Reasons: t 1.55 on the shipped window is the
weakest of the three; the release-day cell (which had the t 2.06) died
post-2018, so the trade leans on the week-long window whose modern-era
support is N=13; and 27 bps of expected edge over six sessions on an
instrument with roughly 1% weekly vol is a thin Sharpe even at futures-level
costs. It shipped because both controls (flat all-NFP weeks, negative
generic washouts) behaved exactly as the thesis requires, the current
washout is deeper than the cell's -1% threshold, and it diversifies the
day's basket away from US equity beta. But if one of the three is noise,
it is this one.

Confidence ordering: idea 2 (XLU/SPY, t -3.15, beats its own controls by 4x)
> idea 1 (Aug-NFP short, hit 35% over 26 years but era-lumpy) > idea 3.

## What I wanted from the state file and could not get

- **Live positions** — STATUS_TOKEN unset, so the broker book was
  unavailable. Overlap statements ("the book is flat") lean on
  staged_signals being empty, the exposure leg at 0x and the trend sleeve
  state reset; actual open OLV/OVS legs from earlier in the week are
  invisible to me. If the primary account holds anything short-vol or
  short-utilities, idea 2's overlap line is incomplete.
- **event_sleeve_state.json** was missing (warning in state). I know from
  the prereg that no event-sleeve window is active until Aug 21 (V4), but I
  had to reason that from documentation rather than state.
- **Positioning/flow data** — nothing in the state supports the
  flow_mechanics axis: no COT, no dealer-gamma estimate, no ETF flow series.
  A CFTC dollar-positioning readout would have directly
  strengthened or killed idea 3's "trend followers pressing the move" claim.
- **Consensus estimates for tomorrow's NFP** (whisper vs prior) — the
  calendar knows the date but not what is priced; both NFP ideas argue
  about positioning into the print without seeing the survey number.
- **Intraday/overnight context for today's session** (deliberately excluded
  by the bake-off timing rule, but a normal 7 AM run would also benefit from
  futures levels vs the prior close when writing MOC entries).
