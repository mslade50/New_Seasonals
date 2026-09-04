# Cell map - run date 2026-08-27 (Thursday)

asof session 2026-08-27 | next session 2026-08-28 (Friday) | midterm year
Sweep: 1245 cells scanned, 98 fired (72 event / 26 price). BH crit p 0.0145, 14 pass.
Cap: P5b:rank21_extreme kept 8, dropped ETH-USD and EURUSD=X (both crowded into the
same ags/crypto stretch story already represented; neither would have published).

## Price-lane availability

The 21:10 UTC price job did not run again tonight (same shed cron as 2026-08-26;
`gh run list` shows no PM run, newest is a 7s AM-fallback short-circuit at 19:53Z).
R2 and the local cache both held the 08-26 bar. Re-ran `scripts/update_master_prices.py`
locally with the four R2_* vars blanked so `cache_io.is_configured()` returned False
and the shared copy was never written. Local cache now at 2026-08-27, price lane live,
`meta.prices_fresh` true. R2 is still stale and that is deliberate; the PM job owns it.

## Standing constraint: what 2026-08-26 already spent

Last night published month-end for SPY/IWM (third-to-last August session), TLT/IEF
(duration leg through Friday and through month end), and the IEF month-end-is-not-a-
Jackson-Hole-effect decomposition. It also published BTC z10 and ZW=F wheat. All five
are off the table tonight except where a genuinely new number is attached.

---

## Event lane

### E:jackson_hole - Jackson Hole on 2026-08-28 (next session), anchor = tonight's close
The whole reason tonight is not quiet. h1 IS the speech session.
Engine cross-asset table, n 23-26, k1:

| subject | h1 mean | record | sign p | t | era stable | BH |
|---|---|---|---|---|---|---|
| ^VIX | -2.61% | 5-21 down | 0.0012 | -1.72 | yes | pass |
| CL=F | +0.69% | 21-5 up | 0.0012 | 1.72 | yes | pass |
| IWM | +0.66% | 21-5 up | 0.0012 | 1.98 | yes | pass |
| HG=F | +0.82% | 19-6 up | 0.0073 | 3.02 | yes | pass |
| SPY | +0.30% | 19-7 up | 0.0145 | 1.32 | no | pass |
| GC=F | +0.57% | 17-8 up | 0.0539 | 2.81 | yes | fail |
| SI=F | +0.62% | 17-8 up | 0.0539 | 1.86 | yes | fail |
| EEM | +0.52% | 14-9 up | 0.2024 | 2.03 | yes | fail |
| ^GSPC QQQ | +0.28% | 18-8 up | 0.0378 | ~1.1 | no | fail |
| TLT IEF ^TNX HYG JPY EUR DX NG | mixed, nothing | | | | | |

VERDICT **DRILL**. Three reasons the engine table cannot ship as-is:
1. **The confound is structural.** 26 of 27 symposium dates in `macro_events.csv` are
   FRIDAYS (only 2020-08-27 is not: virtual, Thursday keynote), and all 27 are in
   August. The bare `E:weekday_month` cell "Fridays in August" has ^VIX at 33-85 down,
   -0.97%, n=119, sign p 0.0000, era stable, BH pass. So a large part of the JH VIX
   number may be nothing but August Friday. Same question for IWM and CL=F. This has
   to be decomposed before a single one of these publishes. -> `01_jh_vs_aug_friday.py`
2. 2013 and 2015 carried no chair speech and are in the sample.
3. Concentration and the follow-on are unmeasured.
Novelty: every JH fingerprint is `is_new`, nothing blocked.
Sweep status: found by the search, owes BH. Not a pre-specified hypothesis, and the
map says so.

### E:month_end - final 3 sessions of the month, anchor = tonight's close
Tomorrow is td 20 of 21, one session from the end. Engine solids: IEF +0.077% t 5.65,
TLT +0.126% t 4.47, HYG +0.065% t 2.57, ^TNX -0.294% t -4.42, all n 700-960.
VERDICT **SKIP(published 2026-08-26)**. This is verbatim last night's items 2 and 3,
including the ^TNX/IEF duration leg and the JH decomposition of it. Re-telling it with
one session shaved off the horizon is exactly the countdown re-telling the novelty rule
bans. The one live question it leaves, whether tomorrow belongs to the symposium cell
or the month-end cell, was answered last night in IEF's favour and is folded into
drill 01 as a control rather than published again.

### E:weekday_month - Fridays in August, anchor = tonight's close
^VIX 33-85 down, -0.97%, sign p 0.0000, BH pass, era stable, n=119. GC=F +0.243%,
t 2.50, 67-48. Everything else inside noise.
VERDICT **DRILL**, but as the CONTROL in `01_jh_vs_aug_friday.py`, not as its own item.
Publishing "August Fridays are soft for VIX" beside "Jackson Hole is soft for VIX"
would be one fact told twice, which the prose rules forbid. Whichever survives the
decomposition is the one that ships.

### E:seasonal_doy - same trading day of year (+/-2), Aug 28
Best cells: IWM h1 18-7 up sign p 0.0216 but mean only +0.02%; HYG h5 15-4 sign p
0.0096 on n=19; TLT h1 16-7 sign p 0.0466. Every midterm sub-cell is n=5 or 6 and
splits 4-2 or 3-3.
VERDICT **SKIP(subsumed and weak)**. The doy cell for Aug 28 is mechanically the same
sessions as the JH and month-end cells already under examination, with a looser anchor
(+/-2 days) and no mechanism, and its medians contradict its means. TLT is additionally
the one fingerprint with publication history (2026-08-17). Nothing here beats what
drill 01 is already testing on the same days.

---

## Price lane

### P2 / P2b:new_52w_low - ^VIX3M, first 52w low in 30+ and 90+ days
^VIX3M closed 17.56, exactly on its trailing-252 minimum, -2.39% on the session.
P2 n=22 h1 +1.04% 14-8 t 2.26 era stable; P2b n=13 h1 +1.35% 9-4 t 1.85 era stable.
Neither passes BH, both sign p ~0.13.
VERDICT **DRILL**. Highest-value cell tonight and the only one that crosses both lanes:
three-month implied vol at a one-year low on the eve of the symposium. The rebound
numbers are weak on their own, so the item has to be built on the state and its
company rather than on a forward mean. -> `02_vol_floor.py`

### P5b:rank21_extreme (bottom) - ^VVIX, 21d return in the bottom 5% of its year
n=221 h1 +0.78% t 2.47 era stable, but hit 51.1% and sign p 0.394.
VERDICT **DRILL**, folded into `02_vol_floor.py` as corroboration of the compression,
with the tail-vs-median split stated. A +0.78% mean on a 51% hit rate is a tail
artefact and the brief must say so if it uses it at all.

### P4:z10_extreme (up) - BTC-USD z10 3.07, ETH-USD, plus ZC=F ZW=F ZS=F CT=F
BTC n=297 t 3.17 BH pass, tag hint solid.
VERDICT for BTC/ETH **SKIP(published 2026-08-26)**. Last night's item 4 was this exact
subject at a deeper cut (z10 above 3, which tonight's 3.07 is still inside) and the
verdict was that the deep bucket does not survive declustering or the era split.
Nothing has moved but the countdown.
VERDICT for ZW=F **SKIP(published 2026-08-26)**, item 5.
VERDICT for ZS=F, CT=F **SKIP(no edge)**: h1 t -0.24 and 0.01 respectively.
VERDICT for **ZC=F DRILL** - see the multi-trigger note below.

### P5:rank5_extreme / P5b:rank21_extreme (top) / P7:up_streak - ZC=F corn
Corn is the only subject in the universe firing FOUR triggers at once: z10 3.38,
5d rank 100.0, 21d rank 100.0, 63d rank 100.0, a 5+ session up streak, closing at a
52-week high, +3.40% today, +18.4% in 21 sessions, volume 2.06x its 63d norm.
Each single trigger is weak (h1 t between 0.6 and 2.1, every sign p above 0.25).
VERDICT **DRILL**. The individual cells are unpublishable and the engine correctly
tags them suggestive; the question worth asking is whether the CONJUNCTION is a
different animal. -> `03_corn_conjunction.py`

### P5:rank5_extreme (bottom) + P6:two_atr_day (down) - KC=F coffee -12.92%
VERDICT **DRILL(integrity first)**. Coffee was last night's named data artefact: its
Tuesday session printed -1.68% in tonight's cache where the previous night's cache had
-11.36%. A -12.92% session in the same series is not believable until the bar is
checked against its own high/low and volume. If it is plumbing it is not a nugget and
not a correction either, since last night already told Scott the series is unreliable.
-> `04_coffee_bar_check.py`

### P6:two_atr_day (up) - PA=F palladium +3.95%
VERDICT **SKIP(no edge)**. n=238, h1 t 0.77, hit 50.0%, sign p 0.53, era unstable,
h5 negative. A big bar with nothing behind it.

### P6:two_atr_day (down) / P4 (down) - USDCNY=X -0.14%
VERDICT **DEAD**. A 0.14% session in the yuan is a 2-ATR day only because the fix
barely moves; h1 mean -0.018% is not a sentence anyone can read.

### P7:up_streak - ^BVSP, JPY=X
VERDICT **SKIP(no edge)**. ^BVSP h1 t -0.05, JPY=X t 0.59, both era unstable, and
^BVSP is a foreign cash index whose streak has no US-session read.

---

## Calendar, next 5 sessions

| session | scheduled | verdict |
|---|---|---|
| Fri Aug 28 | Jackson Hole symposium main session, 10:00 ET; second-to-last August session | DRILL 01, 02 |
| Mon Aug 31 | month end, final August session | SKIP, published 2026-08-26 |
| Tue Sep 1 | first session of September, turn of month | SKIP, nothing fired and September's own seasonal is not a next-session claim |
| Wed Sep 2 | nothing scheduled | none |
| Thu Sep 3 | nothing scheduled | none |
| (Fri Sep 4) | employment report 08:30 ET, 6 td ahead | outside the window, calendar line only |

## Drill queue
1. `01_jh_vs_aug_friday.py` - decompose the symposium cell against August Fridays, in VIX, IWM, CL=F, SPY. Concentration, era, midterm, no-speech years, follow-on.
2. `02_vol_floor.py` - VIX3M at a 52-week low: company, breadth of the compression, what the complex did next, and whether it has ever coincided with the symposium.
3. `03_corn_conjunction.py` - does four simultaneous triggers beat any one of them.
4. `04_coffee_bar_check.py` - integrity of the -12.92% bar before anything is claimed.
