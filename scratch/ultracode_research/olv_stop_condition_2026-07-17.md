# OLV stop-condition study — 2026-07-17

Question (McKinley): OLV keeps stacking signals, stopping out at -1R and
re-buying the same dip same day. Straight price stops look incoherent for a
strategy whose thesis is "hold through low-conviction selling." Find a
condition other than price (volume?) for when a stop-out makes sense vs
holding/adding, or size down and drop stops.

Scripts: `scratch/olv_stack_forensics.py`, `scratch/olv_stop_condition_study.py`,
`scratch/olv_stop_condition_robustness.py`, `scratch/olv_stack_cap_test.py`.
Per-variant trade files: `scratch/olv_stopvar_*.parquet`.

## Ledger forensics (329 OLV trades, 2005–2026, both tiers)

- 57% of trades sit in stacked chains (overlap or ≤3 TD gap). 118 stop-outs.
- 35 same-day stop+rebuy events; 47 stops fired while another OLV leg in the
  same ticker was still open.
- Re-entry after a stop wins 64% at avgR +0.58 — the system sells the dip and
  immediately re-buys it. The churn cost is real, not just aesthetic.
- Multi-trade clusters NET +71.7R (mean +0.96R/cluster). Clustering is OLV's
  winning mode; the loss tail is 2–3 leg chains in trending sectors (the
  thing the sector loss gate exists for), not deep stacks (HE 5-leg +5.0R,
  PFE 4-leg +4.7R).

## Counterfactual method

Engine clone re-run from ledger signal dates on today's adjusted cache
(scale-invariant, relative levels only). Parity vs ledger: medAbsDiff
0.0000R, 100% exit-type match (3/329 trades >0.15R off = vintage drift).
Entries FROZEN; only the exit rule varies. R always denominated in the prod
1.25-ATR unit. Dollars = R x each trade's ledger Risk_flat_750k.

## Variants (totR / avgR / win / PF / worst trade / worst same-ticker chain $)

| variant | totR | avgR | win | PF | worstR | worst chain |
|---|---|---|---|---|---|---|
| prod (1.25 ATR intraday) | 189.7 | 0.577 | 60% | 2.43 | -2.29 | -$8.8k |
| no_stop | 239.1 | 0.727 | 71% | 2.94 | -4.79 | -$22.5k |
| eod_stop (close ≤ stop → MOC) | 215.3 | 0.655 | 67% | 2.58 | -3.63 | -$16.5k |
| vol_stop_10 (close ≤ stop & vol ≥1.0x med20) | 212.2 | 0.645 | 67% | 2.52 | -3.51 | -$16.1k |
| **vol_stop_15 (≥1.5x med20)** | **228.2** | **0.694** | **69%** | **2.78** | **-3.32** | **-$13.5k** |
| vol_stop_20 (≥2.0x) | 227.0 | 0.690 | 70% | 2.71 | -3.68 | -$23.0k |
| vol15 + 2.5 ATR disaster stop | 223.6 | 0.680 | 69% | 2.68 | -3.50 | -$14.5k |
| no_stop + 2.5 ATR disaster | 222.1 | 0.675 | 70% | 2.62 | -3.50 | -$16.9k |
| wide 2.0 ATR intraday | 202.6 | 0.616 | 67% | 2.37 | -3.50 | -$13.8k |

Same-day stop+rebuy churn: prod 39 → eod_stop 18 → vol_stop_15 7.

## What the price action actually says

Of 118 trades that touch the 1.25 ATR stop level (no-stop path):
- intraday touch but close recovers above the level: avg final -0.32R
- close finishes below the level: avg final -1.10R
- breach-day volume: quiet (<1.0x med20) breaches recover best (-0.19 avgR,
  38% pos, n=16); elevated-volume breaches worse (~-0.6 to -1.1). Directionally
  supports the low-volume-dip thesis but non-monotone in the middle buckets;
  the close-vs-touch split is the stronger, cleaner effect.

Decomposition of the +38.5R (vol_stop_15 vs prod): ~+25R from "confirm on the
close instead of the intraday touch", ~+13R from the volume filter on top.
Threshold plateau 1.5x–2.0x on totR (228/227); 1.0x is worse (212) — treat
the volume leg as lower-conviction than the close-confirm leg.

## Robustness

- LOYO on vol_stop_15 diff: full +38.5R, min +22.0R (drop 2021).
- Episode-clustered (83 ticker-chains where outcome changes): mean +0.46R,
  t = 2.44. Drop-best-chain: +32.4R.
- Tier split: Liquid +18.7R (on prod base 60.3R, 76 trades), Overflow +19.8R
  — not a survivorship artifact alone.
- Positive in 13/20 years. Worst years: 2026 YTD -5.5R, 2017 -4.0R.

## The honest negative: 2026

Every loosened variant LOSES in 2026 (no_stop -21.9R, vol_stop_15 -5.5R vs
prod). The exact clusters that prompted this study (LMT -3.97 → -4.95,
LYB -3.02 → -4.70, DBC -2.26 → -4.02 under vol_stop_15) were trending-down
clusters where stopping was right. The frustration episodes are NOT evidence
for the rule; the other 20 years are. 2021 (+16.5R) and 2025 (+7.3R) carry
the most weight.

## Rejected options

- **No stop at all**: best totR but fattest tail (-$22.5k chain, worst -4.8R)
  and maximally exposed to the ledger's survivorship blind spot (delisted
  names absent; overflow tier flattered — standing CLAUDE.md caveat). No.
- **Smaller size + no stop**: matching prod's worst chain requires 0.39x →
  $PnL 225k vs prod 460k. Strictly dominated by conditional stops. No.
- **Per-ticker leg caps (1/2/3)**: cost 7–59R and do NOT shrink the worst
  chain (tail chains are 2–3 legs; deep stacks were winners). No.
- **Wide 2.0 ATR intraday**: keeps the churn structure, less edge than EOD
  variants. No.

## Candidate rule (NOT shipped)

Replace OLV's resting 1.25 ATR STP with an EOD check at ~15:58 (same live
mechanism as OVS EOD-DD): exit MOC iff close ≤ entry − 1.25 ATR AND day
volume ≥ 1.5x trailing 20d median. Quiet closes below the level: hold (time
exit at T+10 still bounds everything). Optional +2.5 ATR always-on disaster
stop as survivorship/blowup insurance: costs ~4.6R/21y, keeps a resting
order in the book overnight.

Expected costs to accept: per-leg tail moves from ~-1R to occasional -2 to
-3.3R; worst chain ~-$13.5k vs -$8.8k flat basis; no resting stop overnight
(gaps evaluated at next close — modeled as such in the sim).

Second-order effects NOT modeled: sector-gate realized-R timing shifts
(later realization → gate trips later → more entries in June-2026-type
clusters), ladder rung drift from longer holds. Both push toward re-running
the full engine before any ship decision, plus the house prereg discipline
(freeze rule + thresholds first; no re-scanning).

## Part 2 (2026-07-18): confirm on settled close, exit NEXT OPEN

McKinley's objection to the 15:58 version: final volume isn't known at 15:58.
Variant `nxo_15`: same confirm (close ≤ entry − 1.25 ATR AND day volume ≥
1.5x med20, evaluated after the close on settled data), exit at the NEXT
session's open (MOO, 3 bps slip). Script:
`scratch/olv_stop_nextopen_test.py`; trades `scratch/olv_stopvar_nxo_*.parquet`.

- nxo_15: totR 224.8 / avgR 0.683 / win 69% / PF 2.70 / $547k — costs only
  3.4R vs the 15:58 MOC version (228.2), spread thin (worst single-year gap
  -1.4R). The whole cost is overnight drift: mean -0.06R per exit, -4.0R
  total across 63 exits (63% gap down). Worst trade actually improves
  (-2.94 vs -3.32); p5 slightly worse (-2.02 vs -1.91).
- The feared exit-morning rebuy churn DOESN'T materialize: only 3/63 exits
  have another OLV leg filling the same day (2 more the day after; 3 exits
  had a fresh signal on the confirm close). STRUCTURAL reason: the exit
  requires a volume SPIKE while a fresh OLV signal requires 10d volume rank
  < 15 — the two conditions oppose each other, so a volume-confirmed stop
  day is almost never simultaneously a valid new entry. The rule is
  self-consistent in exactly the way the old intraday stop wasn't.
- Operationally simpler than the 15:58 leg: decision runs post-close on
  settled bars (fits the PM pipeline), staged as a MOO sell next morning —
  no intraday conditional order, no IBKR-side volume estimation.
- nxo_eod (no volume condition, next-open exit): 211.6R — the volume filter
  still earns its keep in this geometry (+13R).

## Part 3 (2026-07-20): SHIPPED as a package

Follow-on analyses (same thread) before the ship decision:
- Notional cap 50% NAV, OLV single stocks, per-ticker concurrent
  (`scratch/olv_notional_cap_cost.py`, `_nogate.py`, `_nogate_nxo.py`):
  binds 3-8 legs / 21y depending on config, every clipped leg a WINNER
  (balloon stacks are low-ATR names: PFE 85%, MCD 79%, AON, ungated OXY
  61-63%); cost ~4% of OLV PnL. ADOPTED as catastrophe insurance.
- Sector gate: ungated pass shows the gate's live drop list at +10R (it
  blocked the entire late-June oil recovery). REMOVED.
- Ladder: flat 1.0x beats [0.85,1,1.15] ($654k vs $627k) — the 0.85
  first-rung discount was a drag. REMOVED entirely.
- Down-day add-on filter (add only when signal-day ret < 0 ATR): REJECTED —
  up-day add-ons are the BEST segment (+1.09R avg, 78% win, n=27); the
  relationship is U-shaped with the weak cell at mild-down days.
- Final package (no gate, no ladder, nxo_15 exits): 348 trades, totR 245.1,
  avgR 0.704, win 70%, $654k flat vs prod $466k; worst chain ORCL -$17.2k;
  2026 flips -$1.1k -> +$41.5k (one V-bottom episode — discounted).

Shipped 2026-07-20: strategy_config (stop_mode/stop_vol_mult/
ticker_notional_cap + OLV_CAP_EXEMPT_ETFS; ladder + gate fields removed),
strat_backtester (vol-confirm exit branch + notional-cap replay), daily_scan
(Use_Stop=False stamp, load_open_position_notionals + sizing cap,
stage_olv_vol_confirm_exits -> OLV_Exits tab), order_staging/eq_order_entry
(load_olv_exit_rows + Is_Position_Exit naked-MOO exit path, primary account
only). Guards: tests/test_olv_stop_and_cap.py, OneDrive test_olv_exits.py.
CLAUDE.md section: "OLV Vol-Confirmed Stop + Notional Cap (2026-07-20)".

## Part 4 (2026-07-29): OR-composite volume confirm — REJECTED

Question (McKinley): widen the volume leg to "day vol >= 1.5x med20 OR
trailing 3d avg vol (incl. confirm day) >= 1.5x med20"? Intended target:
grinding declines on persistently elevated but never-spiking volume (the
2026 trending clusters). Pre-registered before running: adopt only if diff
totR >= 0 vs nxo_15, or costs <= 5R while materially improving BOTH the
worst same-ticker chain $ and the 2026 tail. Threshold fixed 1.5x, no
scanning. Script: `scratch/olv_stop_or3d_test.py`; trades
`scratch/olv_stopvar_nxo_15_or3d.parquet`.

| variant | totR | avgR | win | PF | worstR | stops | $PnL |
|---|---|---|---|---|---|---|---|
| nxo_15 (shipped) | 243.4 | 0.702 | 71% | 2.84 | -2.94 | 63 | $623.6k |
| nxo_15_or3d | 242.9 | 0.700 | 70% | 2.83 | -2.94 | 66 | $622.7k |
| nxo_eod (no vol) | 220.7 | 0.636 | 68% | 2.52 | -4.02 | 95 | $543.1k |

- The avg3 leg fired only 4 extra confirms in 21y (66 vs 63 stops).
  Structural reason: a decline heavy enough to push the 3d AVERAGE to 1.5x
  med20 almost always contains a single day >= 1.5x, which already
  confirms. The target population ("grinding elevated volume, no spike
  day") barely exists.
- The 4 changed trades net -0.6R (1 better, 3 worse; cluster t = -0.33).
  2026 — the motivating tail — got WORSE (-1.06R). Only 2020 gained
  (+1.0R). LOYO diff min -1.6R.
- Zero tail benefit: worst trade identical (-2.94R), p5 identical, worst
  same-ticker chain byte-identical (ORCL -$17.2k, untouched).

Fails both prereg branches -> rejected. Corollary evidence: the +22.7R gap
to nxo_eod re-confirms the single-day spike condition carries the volume
filter's entire value on this vintage.

Vintage note: baseline totR here (243.4) sits above Part 2's 224.8 because
entries are frozen from the CURRENT post-package ledger (2026-07-21 build,
348 trades, no gate/ladder) on the current cache; the comparison is
internal to the run, so the verdict is unaffected.
