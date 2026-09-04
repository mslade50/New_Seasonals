# Cell map — run 2026-08-11 (Tue), asof session 2026-08-11, next session 2026-08-12 (Wed)

Prices fresh (core bar 2026-08-11). 1,203 cells scanned, 98 fired, 72 event / 26 price.
BH crit p 0.0096, 8 pass. Cycle: midterm. Next session is a CPI print, so the event
lane leads by the top-tier rule.

## Repetition state carried in by hand

`novelty.flags` is empty of blocks, but the engine only tracks its own fingerprints and
last night's brief used four hand-conditioned ones. From `data/context_journal.jsonl`
(2026-08-10) the following are spent:

| published last night | tonight's status |
|---|---|
| `E:cpi\|^VIX\|k2\|spy_at_52wh` — VIX RISES on the CPI eve with SPY at a high | k1 escalation available ONCE, and only with a new conditioning variable. Re-telling with SPY-at-highs again would be a countdown. |
| `E:cpi\|TLT\|k2\|crude21d_hot` — TLT over the print after crude runs | crude is still hot. Same subject, same event, one night later. Needs a genuinely different conditioning or it does not go in. |
| `P9:level_divergence\|SPY_TLT\|high_low` | DEAD tonight. Published, and published as a null (SPY 8-7, TLT 5-10). The state has not moved. |
| `P5:complex_thrust\|SI=F\|gold_silver_pt_pd` | precious-complex thrust is spent. Gold on a CPI print is a different lane and stays available. |
| `E:seasonal_doy\|SPY\|anchor_walk` | the anchor-walk debunk was last night's whole third item. Running it again on tonight's doy cell is the same nugget with a new date. |

## Calendar inside the next five sessions

| entry | verdict |
|---|---|
| CPI Wed 2026-08-12, 08:30 ET (td+1, top tier) | **PUBLISH** — leads the brief. |
| PPI Thu 2026-08-13, 08:30 ET (td+2) | SKIP(second-tier, and it sits behind CPI in the same week; last night already spent a PPI anchor on gas) |
| VIX expiry Wed 2026-08-19 (td+6) | SKIP(outside the five-session window, calendar line only) |
| opex Fri 2026-08-21 (td+8) | SKIP(outside window, calendar line only) |
| Jackson Hole Fri 2026-08-28 (td+13) | SKIP(outside window, calendar line only) |
| FOMC Wed 2026-09-16 (td+25) | SKIP(far) |

## Event lane

### E:cpi — CPI on 2026-08-12, anchor is today, so h1 is the print session itself
n=317 anchors back to 2000. Pre-specified: CPI is a named event the sweep was told to
look at, not a cell the search found, so the BH column is informational here.

| subject | engine | verdict |
|---|---|---|
| ^VIX | n=317, -0.847%, 120-194 down, t=-2.12, edge -1.105, era-stable, BH pass | **DRILL 03** — the strongest cell in the sweep and the print-day mirror of last night's eve cell. Only publishable if a new conditioning variable earns it. VIX enters at 15.28, 50.8% off its 52-week high, 17.7% under its 200d, so the honest question is whether the crush needs premium to crush. |
| ^GSPC / SPY | n=317, +0.021% / +0.029%, edge -0.012 / -0.010, hit 55.8 | **DRILL 01** — the mean is nothing and that IS the nugget. The index has no directional edge on a print; the day is a dispersion event. Needs the dispersion measured, not the mean re-reported. |
| QQQ | n=317, +0.115%, 182-133, sign p 0.0048, t=1.26, BH pass | **DRILL 01** — four times SPY's mean on the same 317 anchors. The QQQ-minus-SPY spread is the cell, not QQQ alone. |
| GC=F | n=309, +0.140%, 176-132, t=2.29, sign p 0.0084, era-stable, BH pass | **DRILL 04** — gold enters z10 +2.14, 5d rank 97.6, 21d rank 90.9. Cross the event cell with the stretch. |
| SI=F | n=309, +0.158%, 177-130, sign p 0.0061, BH pass | SKIP(silver was last night's headline subject in the precious-thrust item; gold carries the metal slot tonight) |
| EEM | n=278, +0.134%, 57.2%, sign p 0.0096, era-stable, BH pass | SKIP(real but thin at +0.13%, and EM tonight is better served by the Brazil item in the price lane, which is live rather than scheduled) |
| ^TNX | n=317, -0.187%, 140-174, t=-1.37, sign p 0.0459 | **DRILL 02** — carried as the yield side of the bonds-at-lows cross. |
| TLT / IEF | TLT n=286 +0.057% t=0.99 era-unstable; IEF n=286 +0.050% t=1.62 | **DRILL 02** — the unconditional cell is nothing. The live fact is TLT 0.33% off a 52-week low, IEF 0.77%, LQD 0.62%. Different conditioning from last night's crude cross, so it clears the repeat rule if the number is there. |
| EURUSD=X | n=269, +0.077%, t=2.18, sign p 0.0564 | SKIP(t is carried by a few large prints and the record is 148-120; the dollar has no live extreme tonight beyond a -1.63 z10) |
| DX-Y.NYB | n=317, -0.037%, t=-1.14, era-unstable | SKIP(era-unstable and economically nothing) |
| HG=F, CL=F | era-unstable, sign p 0.15 and 0.37 | SKIP(no edge, and crude was last night's conditioning variable) |
| JPY=X, HYG, IWM, NG=F | \|t\| < 1 or sign p > 0.05 with no live state | SKIP(nothing there) |

### E:ppi — PPI on 2026-08-13, anchor 2 td out
Best cell is CL=F at +0.322%, t=2.37, era-stable, BH fail. SKIP(all of it) — second-tier
event behind a CPI in the same week, and last night already published a PPI anchor.
Nothing here beats a live CPI cell on relevance, which is the first ranking criterion.

### E:weekday_month — Wednesdays in August
n=117. ^GSPC +0.096% 70-47 sign p 0.021, QQQ +0.185% 68-47, CL=F -0.389% t=-2.06.
SKIP(all) — the bare weekday-by-month cell is the least specific thing in the sweep, and
tomorrow's Wednesday is a CPI Wednesday. The calendar cell is confounded by the event
cell that is already leading the brief.

### E:seasonal_doy — same trading day of year, Aug 12
QQQ 8u-18d sign p 0.0378 all years; ^VIX 9u-17d all years and **0-for-6 in midterm
years, mean -5.18%**; ^TNX 9u-17d, midterm -0.72%.
SKIP(anchor-walk, plus confounded) — last night's own third item showed these doy cells
survive only at the exact anchor the calendar lands on. Worse, mid-August CPI prints
land on Aug 10-14 in most years, so the Aug-12 doy VIX cell and the CPI VIX cell are
substantially the same days. Publishing both would be double-counting one effect. The
CPI version has 317 anchors and a mechanism; the doy version has 6 and a coincidence.
Noted in drill 03 as a confound to check, not as a nugget.

## Price lane (anchored on today's close, 2026-08-11)

| trigger | subjects | verdict |
|---|---|---|
| P5:rank5 top — ^SKEW | n=359, -1.352%, 31.2% hit, t=-8.02, era-stable, BH pass, engine hint `solid` | **DRILL 05** — SKEW's own reversion is near-tautological (a bounded, fast-mean-reverting index), so the engine's cell is not the nugget. The question the engine did not ask is what the S&P does after a SKEW thrust, with a CPI landing the next session. |
| P7b:down_streak — ^BVSP | n=143, +0.406%, 85-58, t=1.87, sign p 0.0147, era-stable | **DRILL 06** — and it agrees with P5:rank5 bottom on the same market. Two independent triggers naming Brazil is the cross worth computing. |
| P5:rank5 bottom — ^BVSP, EWZ | ^BVSP n=314 +0.174% sign p 0.097; EWZ n=313 +0.248% sign p 0.0567 | **DRILL 06** — same item. EWZ -3.44% today, ^BVSP -2.50%, 5d rank 0.8. |
| P5:rank5 bottom — HE=F | n=304, -0.171%, era-stable | SKIP(lean hogs; the universe rule is macro subjects only, and a -12.96% hog session is not context for tomorrow's S&P session) |
| P4:z10 — GC=F | n=286, +0.005%, t=0.06, edge -0.044, era-UNSTABLE, BH pass on sign only | **DRILL 04** as the control side: gold's stretch alone predicts nothing, which is what makes the CPI crossing worth measuring rather than assuming. |
| P4:z10 — USDTRY=X | n=576, +0.196%, t=3.14, 69.1%, BH pass, hint `solid` | DEAD(a managed currency in structural depreciation; the "edge" is the carry drift, and it is not context for a US session) |
| P4:z10, P5, P5b, P7 — SB=F | four separate triggers, best sign p 0.41 | SKIP(sugar fired four triggers and none of them has a forward result; a busy ticker is not a signal) |
| P5:rank5 top — CADJPY=X, GBPJPY=X | \|t\| < 0.6, era-unstable | SKIP(yen-cross momentum with no forward edge) |
| P5b:rank21 bottom — CHFJPY=X | n=334, +0.061%, sign p 0.0163, era-stable | SKIP(3bp mean; statistically alive, economically absent, and no US-session relevance) |
| P5b:rank21 top — SB=F | n=426, +0.044%, 208-211 | DEAD(no effect) |
| P6:two_atr down — HE=F, USDCNY=X, LE=F | HE=F t=-2.77 | SKIP(hogs and cattle again; USDCNY at 43.6% hit with a positive mean is incoherent, sign p 0.92) |
| P6:two_atr up — ZC=F | n=182, +0.108%, 77-97 down, sign p 0.21 | SKIP(mean and record disagree, so there is nothing to say) |
| P7:up_streak — NZDJPY=X, AUDJPY=X, USDHKD=X | NZDJPY t=2.08 on a 0.098% mean; USDHKD moves 0.003% | SKIP(pegged currency and carry noise) |
| P2 / P2b:new_52w_low — USDMXN=X | n=21 and n=13, +0.162% / +0.090% | SKIP(n=21 on a peso low, and the engine's own 90-day version halves the effect; a 16bp anecdote is not worth one of six slots) |
| P8:sma200_cross — HE=F | n=25, era-unstable, t=-0.44 | DEAD(hogs, no effect) |

## Live states the sweep did NOT fire on, checked by hand

- TLT 0.33% off its 52-week low, IEF 0.77%, LQD 0.62%. No P2 fired because none of the
  three actually printed the low today. This is the most specific live macro fact on the
  board and it goes into drill 02.
- SPY 0.35% off its 52-week high with the long end on the floor: published last night as
  a null. Not repeated.
- VIX 15.28 with VIX3M 18.91, term structure steeply normal, no P10 fire. Feeds drill 03
  as the entry condition rather than as its own item.

## Drill queue and outcomes

| script | question | outcome |
|---|---|---|
| `01_cpi_index_dispersion.py` | The index mean is zero on a print. How wide is the day, and is the QQQ-over-SPY spread real? | **TWO PUBLISH.** The width is a well-powered null: mean absolute move 0.848% on 314 prints against 0.800% elsewhere, 1.06x, and P(>1%) 28.7% vs 27.6% at z=+0.43. The spread is real: QQQ over SPY 187-127, sign p 0.0004, control 52.5%, survives trimming the five largest days, and NFP gives 153-157. |
| `02_cpi_bonds_at_lows.py` | CPI prints that arrive with the long end already at a 52-week low. | **KILLED.** TLT n=14, 8-6, sign p 0.40. IEF n=27, 14-13, and its top two episodes carry 104% of the total. Yields 6-8. The state is striking and nothing follows it. Not published, which also keeps TLT-into-CPI from running two nights straight. |
| `03_cpi_vix_low_entry.py` + `09_final_numbers.py` | Does the CPI-day vol crush still happen when VIX enters with no premium? Plus the Aug-doy confound. | **PUBLISH.** VIX enters at the 12th percentile of its trailing year. On the 102 prints that met a bottom-quartile VIX, VIX fell 64-36, sign p 0.0065, against 47.1% on 2,144 bottom-quartile sessions with no print. Cell re-cut in drill 09 so the control matches the published definition exactly. Confound check came back CLEAN: only 3 of 22 mid-August midterm sessions are CPI prints, so the doy cell is not the CPI cell. It stays skipped on the anchor-walk reason alone. |
| `04_cpi_gold_stretched.py` | Gold on a print when gold is already extended, against gold-extended alone. | **PUBLISH as a null inside a positive.** The famous cell is real at 173-132, sign p 0.0128, and it lives entirely in the 267 prints where gold was NOT already run (153-113). The 27 that arrived with gold in the top decile of its own year split 14-13. Gold enters tonight at the 97.6th. |
| `05_skew_spike_spy.py` | S&P forward returns after a SKEW 5-day thrust into the top 5% of its year. | **KILLED on direction.** 190 episodes, h1 edge -0.103% against the local control, h5 +0.110%, h21 +0.097%. The raw hit rates look strong and are drift. |
| `06_brazil_capitulation.py` | ^BVSP down-streak and 5d-bottom-5% together, and EWZ alongside. | **KILLED.** The cross is 43 episodes and October and November 2008 carry 93% of ^BVSP's h5 total. It flips sign across 2018: +1.94% at 58.6% before, -0.42% at 46.2% after. EWZ the same shape. Two triggers naming one market turned out to be one crisis naming itself twice. |
| `07_skew_vix_divergence.py` + `08_skew_or_just_vix.py` | Tonight's exact shape: tail bid thrusting while at-the-money vol falls. | **PUBLISH, downgraded by its own control.** 131 episodes are followed by 13.57% realized vol over the next 10 sessions against a 16.08% local control, but a VIX-decile-matched control gives 14.69%, so the residual is -1.11pp and most of the effect belongs to the falling VIX rather than the tail bid. Published as that correction, not as the raw number. |

## Final slate

Five nuggets, four event and one price, no anecdotes, no cell tagged `solid`.
Headline is the dispersion null: it is the highest-N claim on the board, it is the most
counterintuitive, and it sets up the two items that follow by explaining where the print
does show up. Nothing here inherits a `tag_hint` upgrade, and the only BH-relevant cells
(CPI subjects) are pre-specified, so the sweep correction does not apply to them.
