# Cell map — run 2026-09-09 (Wed)

asof session 2026-09-09 | next session 2026-09-10 (Thu) | midterm year, September
prices_fresh TRUE (core bar 2026-09-09). Both lanes live.
sweep: 1217 cells scanned, 105 fired (72 event / 33 price), BH crit p 0.0057, 5 pass.

## The tape I am writing into

Reflation signature on the eve of back-to-back prints. ^TNX 4.837 AT its 52w high
(z10 1.76), ^FVX at its 52w high (z10 2.24), ^IRX 63d rank 98.4. IEF sits 0.05%
off its 52w LOW, LQD 0.09% off, TLT 0.85% off. Commodities: HG=F, ZC=F, ZS=F,
ZW=F, SB=F all at 252d highs; CL=F +3.91% (z10 2.73, 21d +16.2%). Equities soft
and narrow: SPY -0.46, QQQ -0.29, ^DJI -0.77, IWM -1.37; ^VIX +4.71 to 16.46,
^VVIX +6.55. Outliers: KC=F -8.57%, NG=F -3.70%, SB=F +6.91%, CT=F +5.36%.
Breadth 59.8% above the 200d, from 64.4% 21d ago.

## Calendar verdicts (next 5 sessions + the near shelf)

| date | td | event | verdict |
|---|---|---|---|
| 2026-09-10 | 1 | **PPI** | DRILL. Next session, and it lands the session BEFORE CPI, which is the inverted ordering. That inversion is the cell, not the PPI base rate. |
| 2026-09-11 | 2 | **CPI** (top tier) | DRILL, narrow. 2026-09-08 already published the Friday-CPI print cell and the k3 run-up. A second CPI telling at a shorter countdown with the same content is the banned repeat. Only publishable if the drill produces the PPI-then-CPI pairing, which is new. |
| 2026-09-16 | 5 | FOMC decision | SKIP(too far, and the pre-FOMC drift window opens 4 td before, i.e. 09-10 is td-4). Note in calendar block only. |
| 2026-09-16 | 5 | VIX expiry | SKIP(no distinct cell fired; folded into the FOMC date). |
| 2026-09-18 | 7 | opex / quad witching | SKIP(outside the 5td window; calendar block only). |
| 2026-10-02 | 17 | NFP | SKIP(far). |

## Trigger-group verdicts (all 15 in cells_index)

### Event lane

1. **E:ppi** (18 subjects, k1) — **DRILL**. IEF is the only `solid` hint in the
   whole sweep and one of the 5 BH passes: n=287, +0.062%, 58.2% hit, t=2.67,
   167-117, sign p 0.0033, era-stable. TLT n=287 t=2.42 sign p 0.009 is the same
   effect at longer duration. Base rate is real but the mean is a rounding error;
   it only earns space if it survives today's conditioning (bonds entering AT the
   52w low, 10y AT the 52w high). Drill 02. Pre-specified? No: the PPI-day bond
   bid was found by the sweep, so it owes BH, and it passes.
   ^TNX k1 (43.4% hit, 138-177, sign p 0.0248) is the same coin, do not double-tell.
2. **E:cpi** (18 subjects, k2) — **SKIP(published 2026-09-08 at k3; countdown
   re-telling banned)** as a standalone. The only route back in is the ordering
   drill (01), where CPI appears as the second leg of a pair, not as its own cell.
3. **E:weekday_month** — Thursdays in September, n=111. Equity legs are noise
   (^GSPC +0.008%, SPY -0.018%). JPY=X 70-40 up, sign p 0.0027, BH pass is the
   one live number. **SKIP(subject collision)**: the yen complex was the 09-07
   headline (breadth-7) and the 09-06 nugget, and a bare weekday x month cell is
   the weakest possible frame to retell it in. ^VIX 47-64 down, edge -0.60%,
   t=-0.58 — **DEAD** on t.
4. **E:seasonal_doy** (Sep 10, +/-2) — **DRILL**. HG=F midterm h1 is 0-for-6,
   mean -0.771%, sign p 0.0156, and copper closed today AT a 252d high on a
   5-session run. The crossing is the point (drill 05). Also noted: CL=F midterm
   5-of-6 down, mean -1.384%; ^TNX midterm 5-of-6 up +1.355%. TLT and NG=F are
   `repeat_blocked` (published 09-03 and 09-08) — **SKIP(novelty)**.

### Price lane

5. **P4:z10_extreme, stretched down** — KC=F z10 -2.16 with an -8.57% session:
   **DRILL** (06). The four yen crosses (NZDJPY -2.82, CHFJPY -2.64, GBPJPY
   -2.48, CADJPY -2.13): **SKIP(published 2026-09-07 as the breadth-7 cell and
   2026-09-06 as the TNX/USDJPY cross)**. Nothing has moved enough to re-tell.
6. **P4:z10_extreme, stretched up** — CL=F z10 2.73 on +3.91%: **DRILL** (03,
   as part of the commodity complex rather than alone; its own cell is
   t=-0.09, n=172, which is **DEAD** as a standalone claim). ^FVX z10 2.24
   feeds drill 03 as the rates leg. ZS=F, AUDNZD=X: **SKIP(weak, t<|0.2|)**.
7. **P5:rank5_extreme, bottom 5%** — KC=F (rank 1.2) into drill 06. JPY=X
   (n=329, 59.0%, t=2.11, BH pass) and the five yen crosses:
   **SKIP(novelty, same as 5)**.
8. **P5:rank5_extreme, top 5%** — AUDNZD=X only, t=-0.97, edge -0.027%.
   **SKIP(no edge)**.
9. **P5b:rank21_extreme, top 5%** — ZC=F 21d rank 100 (+20.5% in 21d, at a 252d
   high), ZS=F 99.2. **DRILL** into 03: the grain complex is half the reason the
   commodity-highs count is what it is tonight. Standalone ZC=F t=2.10 n=427 is
   publishable but generic; the joint cell is the better version.
10. **P5b:rank21_extreme, bottom 5%** — ^FCHI 21d rank 4.0, n=401, 230-171 up,
    sign p 0.0019, BH pass, era-stable. **DRILL** (04). Yen crosses again:
    **SKIP(novelty)**.
11. **P6:two_atr_day, down** — ^FCHI (n=64) and KC=F (n=54) both feed drills 04
    and 06. USDCNY=X 43.8% hit / sign p 0.922: **DEAD** (sign wrong-way vs mean).
12. **P6:two_atr_day, up** — SB=F +6.91% (n=49, t=-0.29) and CT=F +5.36%
    (n=93, t=0.36). **SKIP(no edge either way)**. Softs are noted as tape colour
    in drill 03's breadth count, not as their own nugget.
13. **P7:up_streak** — HG=F 5 straight up closes into drill 05. Its own cell is
    t=0.02: **DEAD** standalone, useful only as the state that crosses the
    seasonal. AUDUSD=X: **SKIP(t=-0.10)**.
14. **P7b:down_streak** — KC=F, n=187, t=0.22, edge -0.006%. **SKIP** as a cell;
    the -8.57% session (drill 06) is the real event, the streak is incidental.
15. **P8:sma200_cross, crossing down** — ^FCHI, first in 63+ sessions, n=20,
    13-7 UP at h1 against a -0.064% mean. **DRILL** (04). n=20 caps this at
    `suggestive` at best and the mean/record disagreement has to be stated.

## Dropped by cap — recheck?

P4 dropped CHFJPY=X, EURJPY=X; P5 dropped EURJPY=X; P5b dropped JPY=X, EURJPY=X.
All yen crosses, all inside the group I am skipping on novelty. Nothing lost.

## Engine hints I am not inheriting

- `tag_hint` "solid" appears once (E:ppi|IEF). I will only ship it at `solid` if
  drill 02 shows the conditioned version does not fall apart, otherwise it is
  downgraded, never upgraded.
- `bh_pass` true on 5 cells: E:ppi|IEF, E:weekday_month|JPY=X, P5|JPY=X,
  P5|CADJPY=X, P5b|^FCHI. Every one was FOUND BY THE SWEEP, none is a
  pre-specified famous hypothesis, so all owe BH and all have it. The
  pre-FOMC drift window (09-10 is td-4 of it) IS pre-specified and would be
  exempt, but it is not a cell I am publishing tonight.

## DATA INTEGRITY — run before selection, and it changed the selection

`08_roll_seam_check.py` + `09_futures_data_integrity.py`. 2026-09-08's brief found
the same complex was printing continuous-contract roll seams rather than prices, so
the test was repeated rather than assumed. It fires again tonight.

| ticker | 2026-09-09 | verdict |
|---|---|---|
| KC=F | -8.57% | **SEAM**. Volume ran 35, 49, 60, 39, 16, 8, 8, 18, 16, 1, 0 through 09-08, then 20,045. Coffee does not trade 8 lots a day. Gapped -9.55%, never traded back through the prior close. |
| CT=F | +5.36% | **SEAM + CORRUPT**. Volume 6 then 0 then 18,883; the `Open` field is literally 0.0000. |
| ZC=F | +3.38% | **SEAM**. Volume decayed 200,811 -> 240 over three weeks, then 234,291. The Sep-to-Dec corn roll, gap +4.21%, no trade back through. |
| ZS=F | +0.48% | **SEAM**. Volume 26 -> 100,017. |
| SB=F | +6.91% | **CORRUPT**. Low 19.07 sits ABOVE Open 18.17, which is impossible. |
| CC=F | +0.87% | SEAM (volume 0 -> 14,853), but small and unused anyway. |
| GC=F | +1.21% | Migration in progress (volume 666 median -> 161,446) but the bar spans the prior close and the gap is +0.12%. Level usable, not used. |
| HG=F | +1.68% | **CLEAN**. Gap +0.56%, low 6.7385 <= prior close 6.739 <= high 6.894. |
| SI=F | +2.46% | **CLEAN**. Spans, gap +0.22%. |
| CL=F | +3.91% | **CLEAN**. 298k against a 232k median and continuous liquidity for weeks. An ordinary gap-up session. |
| NG=F | -3.70% | **CLEAN**. 1.2x median. |

Consequences, all binding:
- **Drill 06 (coffee) is KILLED.** The largest move on the board is not a move.
- **Drill 03's premise is dead as written.** Three of the five "commodity at a 252d
  high" prints are seams. Rewritten to a crude-plus-rates cell using clean
  instruments only, with a per-trigger spanning assertion inside the script.
- **My own tape summary above was wrong about ZW=F**: it is 4.99% BELOW its 252d
  high, not at one. Corrected here rather than quietly.
- Cash indices, rates, vol and FX have no roll concept and are unaffected. Every
  nugget I ship tonight comes from those, plus crude if it earns it.
- Volume zeros on the most recent session are a separate cache artifact (the PM
  update writes a 0-volume bar) and are not by themselves evidence of a roll.

## SECOND FINDING — the seasonal slot is not safe to fill from my own rebuild

`12_tnx_seasonal_doy.py` reconstructed the Sep-10 doy cell on ^TNX, ^GSPC and IEF
and does NOT reproduce the engine. On ^GSPC h1 the engine reports 13 up / 13 down,
mean +0.041%, sign p 0.5775; my rebuild gets **20 up / 6 down, mean +0.285%, sign
p 0.0047** on the same nominal n=26. On ^TNX midterm the engine reports 5-1 up at
+1.355%; my rebuild gets 3-3 at +0.293%.

The cause is not jitter, it is a different definition. The engine's cell is "same
**trading day of year** (+/-2)", i.e. the Nth trading session of each prior year.
Mine anchors on the **calendar date** Sep 10. Sep 10 2026 is roughly the 172nd
trading day, and in a year with a different holiday pattern the two land on
different sessions. Both are defensible; they are not the same cell.

Consequence: I will not publish a Sep-10 seasonal nugget tonight. Quoting the
engine's number while pointing at my script would misattribute it, and publishing
my 20-6 as though the sweep found it would launder a construction I built after
seeing tonight's tape into a swept result. The seasonal slot goes empty and the
`E:seasonal_doy` group's verdict is downgraded from DRILL to
**SKIP(definition mismatch, recorded above)** for every subject including copper,
unless drill 05 reproduces the engine exactly and says so.

## THIRD FINDING — my PPI premise was wrong, and the correction is the nugget

`01_ppi_before_cpi.py`. I assumed PPI-before-CPI was the rare ordering. Pooled over
2000-2026 it is the **most common** one: 133 of 319 prints, 41.7%, against 28.2%
for PPI after CPI. What is true is that it USED to be normal and now is not. BLS
resequenced around 2019: 12 of 12 inverted in 2011 and 11 of 12 in 2017, then 1 of
12 in each of 2021, 2022 and 2023, and 13 of 91 from 2019. The last one was
2025-09-10 into a 09-11 CPI, this week's pattern exactly one year ago.

A sentence pairing "rare" with the pooled 41.7% would contradict itself, so the era
has to be quoted, not the pool.

Second correction: the inverted PPI session does NOT behave differently. Inverted
n=133 against normal n=186, no subject clears a Welch t of 1.3 on ^GSPC, ^TNX, IEF
or ^VIX, and the reversal rate into the CPI session is 69 of 133, a coin flip. The
publishable result here is the NEGATIVE one, which is worth Scott's time precisely
because the setup looks like it should matter.

`02_ppi_bonds_at_lows.py` reproduced the base cell field for field (IEF n=287,
+0.062%, 58.2%, t 2.670, 167-117, sign p 0.0033) and then found the conditioning
splits the record from the mean: entering at the 252d low the record IMPROVES to
15-7 and 68.2%, while the mean collapses to +0.009%, below its own local base rate
of +0.012%, because two sessions (2016-12-14, 2022-06-14) cost more than the other
twenty gained. ^TNX-at-its-high is the same shape at 11-4 and 73.3% on n=15. Both
are `suggestive` at best and the concentration is not a caveat, it IS the finding.
Tonight sits in that bucket on IEF (+0.053% off the low), TLT (+0.852%) and ^TNX
(exactly at its 252d high), and misses the 21d-rank version by a knife edge, the
rank printing exactly 25.00 against a `< 25` rule.

## FOURTH FINDING — drill 03 dies, and the front end needs a ZIRP cut

`03_commod_highs_yields_high.py` (rewritten) returns nothing publishable, which is
the correct outcome rather than a failure. Two reasons, both disqualifying:
- **Tonight does not even qualify for the strict cell.** Crude's 21d return is
  +16.19% but its 252d RANK is 84.52, below the 90 decile cut. The strict version
  is a historical analogue, not tonight's state.
- The strict cell is n=12 with sign-flipping eras and top-2 concentration above
  100% on five of twelve rows; the loosened cell contains tonight but its ^GSPC
  h=1 era split runs +0.325% at a 76.9% hit before 2018 and -0.492% at 35.0%
  after, with top-2 at 95%.
Pairing the legs never amplified anything, it just cut n by 4-10x. The single
strongest number, ^TNX-at-a-252d-high alone (^GSPC h1 -0.340%, t -2.26, n=54), is
the cell that PUBLISHED ON 09-06, so it is barred on novelty anyway. **SKIP.**

Also logged: `pitch_lab.zscore` puts CL=F at z10 **+1.41**, not the +1.41-vs-2.73
gap the tape block shows, because `_metrics_for` scales a 10d return by 21d vol
and `pitch_lab.zscore` uses the trailing-252 mean and sd of 10d returns. Neither
is wrong; they are different statistics and must never appear in one sentence.

`07_front_end_repricing.py` is the best work of the night, and its finding is a
METHOD finding first. A 63-day PERCENT change in a near-zero bill yield is a
divide-by-small-number, so 2009-2015 and 2020-2021 manufacture rank>=95 readings
out of noise: **30 of the 48 headline episodes have ^IRX below 0.50**. The honest,
rate-regime-comparable sample is **18**. On that clean cut:
- **^VIX h=5: 1 up 17 down, hit 5.6%, mean -3.246%, t -3.56, sign p 0.0001, both
  eras the same sign (0.0% and 10.0% hit), top-2 concentration only 39%.** The
  single cleanest cell in the entire evening.
- ^GSPC h=5 +0.709%, t +2.58, 12-6, both eras positive, edge +0.553pp vs all days.
- Every h=21 row is fragile and none is used: ^GSPC h=21 flips era sign and loses
  to its local control by -1.055pp; ^VIX h=21 has 2020-02-04 alone at +146.85%.
Sample skew to be stated in the brief: 9 of the 18 clean episodes fall in
2025-2026, which is structural, since only 2003-2007 and 2025-2026 are non-ZIRP
rate regimes inside this window. n=18 caps it at `suggestive`.

## Planned drills

| # | script | question |
|---|---|---|
| 01 | `01_ppi_before_cpi.py` | How rare is PPI landing the session immediately before CPI, and does that PPI session behave differently from a normal one? |
| 02 | `02_ppi_bonds_at_lows.py` | Does the BH-passing PPI-day bond bid survive when bonds enter at a 252d low / the 10y at a 252d high? |
| 03 | `03_commod_highs_yields_high.py` | Sessions with N+ commodity futures at 252d highs while ^TNX is at a 252d high: forward SPY, ^TNX, ^VIX. |
| 04 | `04_cac_200d_break.py` | ^FCHI first 200d break in 63+ sessions while SPY is within 3% of its own 52w high: forward ^FCHI and SPY. |
| 05 | `05_copper_seasonal.py` | The Sep-10 midterm copper cell (0-for-6) crossed with copper entering at a 252d high on a 5-session run. |
| 06 | `06_coffee_crash.py` | KC=F sessions of -8% or worse: what follows, and is the current 5d -14.9% context normal. |
| 07 | `07_front_end_repricing.py` | ^IRX 63d change at the 98th percentile with the S&P within 3% of its high: what the equity tape did next. |
