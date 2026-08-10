# Cell map — run 2026-08-09 (Sun), asof Fri 2026-08-07, previewing Mon 2026-08-10

Sweep: 1,291 cells scanned, 121 fired (54 event / 67 price). Prices FRESH
(core bar 2026-08-07). Novelty: first run, `delta_suppressed: true` — no NEW
or first-time claims tonight.

## The session being written about

Friday's payrolls printed **-23k against +80k consensus** (an outright job
loss) with unemployment at **4.1% vs 4.2% expected**. The two halves of the
report disagree, and the tape took the dovish read: SPY +0.61%, QQQ +1.17%,
IWM +1.11%, ^GSPC closing AT its 52w high, gold +3.76% (5d +8.7%, 98.8th
percentile), DXY z10 -1.92, VIX -1.65%. Bad news traded as good news.

That configuration is the spine of tonight's brief and it decides most of the
verdicts below: cells that speak to "weak payrolls with equities already at
highs" get drilled, cells that are true of any Monday get skipped.

## Event lane

| trigger | verdict |
|---|---|
| `E:cpi` (3 td ahead) | **SKIP** — the cell is "the 3rd session before a CPI", which is not a thing that happens to a market. QQQ carries p=0.0123 and BH pass on a 179-134 record, but the mean is -0.016% and the edge -0.064: a hit-rate tilt with no magnitude. CPI belongs in the Calendar block on Monday and becomes a real cell on Tuesday evening at k=1. |
| `E:weekday_month` (Mondays in August) | **DRILL** — QQQ n=115, +0.146%, hit 66.1%, record 76-39, sign p=0.0004, BH pass. A two-thirds hit rate on a bare calendar cell is worth a look, but the mean is small relative to the hit rate, which usually means a few large losers. ^VIX +2.34% (t=2.24) on the same cell needs a median before it can be quoted at all. -> `03_august_mondays.py` |
| `E:seasonal_doy` (same doy, Aug 10) | **SKIP** — n=26 all-years, n=6 midterm, and the numbers are noise in both (SPY -0.03% all years, +0.21% midterm; 13 down of 26). A cell that says "coin flip" is not a nugget. Logged so the map shows it was read. |

Calendar entries inside 5 td, each with a verdict:

- Wed 2026-08-12 CPI (+3 td) — **PUBLISH in Calendar block only**, no cell.
- Thu 2026-08-13 PPI (+4 td) — Calendar block only.
- Wed 2026-08-19 VIX expiry (+8 td), Fri 2026-08-21 opex (+10 td), Fri
  2026-08-28 Jackson Hole (+15 td) — outside the 5 td Calendar window and
  outside the 3 td cell window. Not published. Jackson Hole becomes a cell in
  the week of 08-24.
- Month-end: next session is the 6th of 21 sessions in August, 15 from the
  end. No month-end or turn-of-month cell fires, correctly.
- Holiday adjacency: none. Next closure is Labor Day, 2026-09-07.

## Price lane

| trigger | verdict |
|---|---|
| `P12:nfp_below` | **DRILL, headline candidate** — the most specific thing that happened. IWM n=68, +0.32%, hit 66.2%, 45-23, p=0.0052, BH pass; QQQ +0.24%, hit 67.6%, 46-22, p=0.0025, BH pass. But ^VIX +3.26% on the same cell alongside equities up is internally odd and smells tail-driven. Needs era split, concentration, and the cross with "index already at a 52w high". -> `01_nfp_below_at_highs.py` |
| `P12:unemployment_rate_below` | **DRILL** — TLT -0.165%, hit 35.7%, 25-45, p=0.0112, BH pass, and ^VIX +2.74% t=2.96 tagged solid. Economically coherent (a tighter labour print lifts yields, hurts duration) and it points the OPPOSITE way from the payroll half of the same report. The disagreement is the story. Folded into `01_`. |
| negative payroll prints | **DRILL** — not a fired trigger; the engine conditions on the above/below LABEL, not on the sign of the actual. An outright job loss is rarer and more specific than "below consensus". Small N by construction, anecdote at best, but maximally relevant. -> `02_negative_payrolls.py` |
| `P5:rank5_extreme` | **DRILL** — ^NDX n=606, +0.236%, t=2.53, BH pass, tagged solid. The trigger is TWO-SIDED (top or bottom 5%) and the pooled mean hides which side carries it. ^NDX sits at rank 96.0, so only the top-side half is live. If the pooled number is a bottom-side rebound effect the cell says nothing about Monday. -> `04_rank5_which_side.py` |
| `P5b:rank21_extreme` | **SKIP** — same two-sided weakness with weaker numbers, and the fired subjects (FXI, USDTRY, JPY crosses, ^AXJO) are not what a US macro reader needs on a Monday. USDTRY at n=681 t=2.57 is a carry-currency artifact, not context. |
| `P4:z10_extreme` | **SKIP** — USDTRY=X is tagged solid (n=634, t=3.24) and is the same carry artifact: a currency that trends one way for structural reasons will always show a positive conditional. ^GDAXI / ^FCHI / ^AXJO all sit at t under 0.6. Nothing here. |
| `P6:two_atr_day` | **SKIP with one note** — HE=F (lean hogs, -14.0% on the day, the largest move on the tape) is tagged solid at t=-2.82, n=249. Real, but hogs are not macro context for Scott and the brief has better material. Noted so the map shows the biggest single move was seen and dismissed on relevance, not on statistics. |
| `P7:up_streak` | **SKIP** — ^FCHI 5 up closes, next session -0.159%, t=-2.04, 89-115. Borderline, and a French index streak does not earn a slot over the payroll cells. SB=F is noise (t=1.44, 92-86). |
| `P8:sma200_cross` | **SKIP** — ^BVSP n=39 and HE=F n=53, both with |t| under 0.6. DEAD on strength, not on relevance. |
| `P1/P1b:new_52w_high` | **SKIP as fired, but see below** — the only firing subject is USDHKD=X, a pegged currency at the top of its band. Meaningless. |
| ^GSPC at its 52w high | **DRILL as a conditioner** — the index closed AT its 52w high (`dist_52w_high_pct` 0.0) and P1 correctly did NOT fire, because the novelty filter suppresses a state that has been present inside 30 days. That is right for "is this news" and wrong for "is this true", and it is true: it is the condition under which the payroll cell has to be read. Used as a filter in `01_`, not published on its own. |
| gold thrust | **DRILL** — GC=F +3.76% on the day, 5d +8.7%, rank_5d 98.8, z10 1.53, and still 17% below its own 52w high. Not a fired trigger (rank5_extreme capped at 8 subjects and gold missed the cut on |session move| ordering — `sweep.dropped_by_cap` names it). The cap dropped something that matters, which is exactly the case the cap logging exists for. -> `05_gold_thrust.py` |

## Capped cells, recovered

`sweep.dropped_by_cap` lists 7 subjects dropped from `P5:rank5_extreme`. Read:
GC=F is among them and is drilled above. The other six (^AXJO, ZC=F, SB=F and
three JPY crosses) are checked and dismissed on relevance.

## Publication plan

Headline: the payroll cell, because it is the most specific and the most
relevant, and it is the one thing Scott will be thinking about on the drive
in.

- Tomorrow's tape: `01_` payroll cross (2 nuggets: the equity read and the
  duration read), `03_` August Mondays.
- Today in context: `05_` gold, `04_` NDX rank extreme, `02_` negative
  payrolls if it survives.
- Tag discipline: `02_` will be anecdote-grade at best. Budget allows two
  anecdotes and forbids an anecdote leading. Headline stays with `01_`.
