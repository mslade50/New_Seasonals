# Surface map, 2026-08-26 (Wednesday, midterm year)

Freshest bar 2026-08-25. Pipeline all green (7/7), prices/dial/put-call all
dated 2026-08-25. No state warnings. Scoreboard is 4 graded ideas lifetime,
so the per-axis split is not yet readable and is not used to allocate slots
this morning.

Regime: SPY -1.54% off its 52w high, +8.40% above its 200d. Fragility
ma10-63d **88.9** (raw 63d 89.9, raw 21d 69.4), one of the highest readings
in the 2016+ series. Equity P/C fear OFF at the 51st percentile. Signals on:
Defensive Leadership, Low Absorption Ratio. Exposure leg killed (raw 21d
69.4 > 50). Book staged 5 signals (OVS shorts DIS/MRK/HTGC/MSTR, OLV WERN).

This is the tenth morning after nine consecutive stand-downs, so the priority
is surfaces that have never been opened rather than a tenth pass over the
sector cross-section.

Stale-data note: `data/iv_history.parquet` stops at 2026-08-11, so no
candidate may lean on implied vol. The seasonal board payload is stamped
2026-08-05 and its put/call line is three weeks old; it is read as regime
context only.

---

## 1. Calendar events x asset classes

Live window (-5, +15 td): vix_expiry -5, opex -3, **jackson_hole +2**,
**nfp +7**, ppi +10, cpi +11, fomc_decision +14, vix_expiry +14. Month-end
(Aug 31) is **ME-3**.

The registry closed the macro-anchor lane on 2026-08-24 and again on
2026-08-25 ("the inventory of anchors available in late August of a midterm
year is now documented as exhausted"). Two things changed overnight and both
get a verdict rather than a dismissal: **NFP moved from +8 td to +7 td**, so
a hold entered today can reach it inside the 10 td cap for the first time,
and **the month turn is now inside every horizon**.

### jackson_hole, +2 td: CLOSED on every class, not re-opened

Swept on rates, gold, FX, small caps, large caps, credit and international =
seven classes. Ladder 9-for-9 plateau, true offset 8 of 16 at h=10. The
unconditional August tdom 6-16 window beats the anchor outright (+0.234% over
286 starts against +0.102% over 26). Midterm years invert it to -1.485% at
h=10, a sixth independent reproduction. Placebo permutation over relocated
anchors gives P(max-of-6 >= observed) = 0.286 at h=5. Energy is the one class
where the JH-6 anchor survived its ladder, and it is watchlist-parked on
concentration; that anchor was 2026-08-20, five sessions gone, and XLE at
-2.65% off its high sits inside the 5% band the entry forbids. Metals, sectors
and volatility are **not examined** on this anchor: the permutation null is
computed over the family, so an eighth class is exactly the "eighth class on a
dead anchor" the registry names as the wrong move.

### nfp, +7 td: CHECK -> C12

The registry's dismissal was explicitly "+8 td sits at the horizon cap"; at
+7 td a hold entered today reaches the print for the first time. The
post-print direction cell is closed (NFP close to next CPI +0.129% over
N=309 against an all-days control of +0.221%). The ANTICIPATION window
entered seven sessions early has never been measured here. C12 runs it on a
vehicle grid (SPY, TLT, UUP, GLD) so no single class gets a bespoke pass.
Not separately examined: metals (no mechanism connecting payrolls to a miner
thrust that C3/C4's dollar leg does not already carry) and energy (crude's
macro-print cells died on their own clock, 81% of PPI excess accruing after
the number is public).

### ppi +10, cpi +11, fomc +14, quad witching +16, vix_expiry +14

Out of reach on every class. A PPI hold entered today MOC exits ON the print
day, capturing none of the reaction; the rest are beyond the 10 td cap.

### month-end, ME-3: CHECK -> C9, in its September-turn form only

The ME ENTRY anchor is closed on equities (08-24), suspended on rates
(watchlist #12, whose own second condition is literally "NOT August") and
closed on FX (08-25, ME-0 pays -0.55 bp at a 45.6% hit and flips positive
post-2020). What is not closed is the crossing INTO September, which is a
month-of-year object rather than a month-position one. Credit, gold, metals,
energy and international are **not examined** on the month-end anchor: none
has a month-end mechanism independent of the rates leg already suspended.

### earnings: NVDA prints tonight. DISMISSED with numbers

The pre-print side is closed twice, long (2026-08-14) and short (2026-08-19).
The short kill's own placebo ladder puts its peak at k=+4, i.e. the
POST-print offsets, at +2.963% against a true-anchor -0.134%, and that entry
identifies the peak as late-August month position rather than the print. An
index-level version inherits the identical confound. The big-box retail
earnings cluster is also closed (the earnings anchor was worth 1.7 bps
against 18 bps of cost).

Event-anchored candidates generated: **C9 and C12**.

---

## 2. Tape extremes by class (218 names, sorted; 00_tape_sort.py)

| class | today's outliers | verdict |
|---|---|---|
| us_large | SPY -1.54% off high, rank63 19.8; QQQ rank63 10.3, -4.64% off high; DIA rank63 62.3 | the index cross-section is unremarkable. Vehicle for C1/C6/C8, not a trigger. |
| us_small | IWM rank21 42.5, rank63 16.3, -1.92% off high | no standalone extreme; vehicle for C2/C9. |
| rates | TLT **rank5 96.8** (+2.22% in 5d), rank21 57.5, +2.61% off its 52w low; IEF rank5 84.5; ^TNX -1.42% in 5d, at 97.77% of its trailing-252 high | a duration RALLY off the floor. Kills watchlist #18 (needs a +1.5% single day, got +1.10%) and #5 (needs TLT within 0.5% of its 52w low, it is +2.61% above). Feeds C10 and C11 as a conditioner. |
| credit | **HYG at its 52w high exactly (0.00%)**, rank21 92.9, z10 +0.93; LQD +1.36% off its 52w low | credit is leading and equities are not. -> **C6**. |
| gold/miners | **GDX rank21 100.0**, +18.63% in five sessions; **NEM rank21 100.0 and at its 52w high**; GLD rank21 97.2 but -13.68% off its high | the largest thrust in the tape. -> **C5** (breadth form). The GDX fade is closed (08-25, wrong-signed at all ten horizons); watchlist #3's long form is blocked by GLD's own drawdown. |
| other metals | **FCX rank5 100.0, at its 52w high**; SLV +17.74% over 21d but -40.98% off its high; XME rank21 88.1 | copper's own thrust cell was killed 08-24 (reference-class rank 23 of 29). Silver against gold is closed three times. Both fold into C5's breadth count rather than being pitched alone. |
| energy | USO **-4.58% in one session**, rank63 18.7; XLE -2.65% off its high, rank5 18.3; OIH rank63 6.7, OIH-XOP 63d spread at the 2.8th PIT percentile | "long energy at a fresh 52w high with crude at a 63d floor" is closed as an inverter (-1.465%). Watchlist #24 is live in state but its trigger is a RECORD four wins away, which today cannot move. Energy z10 count is 0 of 11, so watchlist #22 is not live. No energy candidate. |
| dollar/FX | **UUP and DX-Y.NYB both at a 21d rank of 0.8** -- the most oversold dollar of the trailing year | the two parked entries built on this (#14, #16) both need a RATE leg that is absent (TNX 21-session change -0.002pt against a +0.20pt floor). Nobody has tested the bare parent. -> **C3**, and its translation **C4**. |
| international | EEM rank63 2.4 with rank5 79.8; EFA at its 52w high; FXI rank5 68.7 | the country-decoupling family is closed on five members (EWZ twice, FXI, SMH/QQQ, EWJ) with permutation nulls. EEM appears only as C4's vehicle, where the mechanism is the dollar rather than the country. |
| volatility | VIX 15.45 (rank21 15.5), VIX3M/VIX 1.1786 = **60.7th pctile of the trailing year**, MOVE/VIX at the **81.7th** | term structure is NOT extreme, so the 98th-percentile contango corpse is not live and is correctly left shut. The MOVE/VIX ratio is the one unexamined vol object. -> **C11**. |
| sectors | XLV rank63 99.6 (closed 08-25); XLF rank63 96.8 at its 52w high; **KRE rank5 9.1, z10 -1.10**; XLI rank5 6.0 rank21 11.1 (closed 08-25); XLU rank21 3.2 (dead in six expressions); **XLRE z10 +1.25 with rank21 only 23.8**; XBI at its high; SMH rank63 0.4 | -> **C7** (KRE against XLF) and **C10** (XLRE, zero registry mentions, snapping up out of a 21d trough on the duration rally). SMH's outright laggard form is **DISMISSED with numbers**: the registry closed it as a regime bet, and today is outside its sample -- trigger days sitting >=15% above the 200d are 4 of 347 and decluster to one episode, while SMH sits +18.53% above its 200d. |
| positioning (CBOE put/call) | **index P/C 10d-MA at the 7.1st percentile of its trailing year while equity P/C sits at the 51.2nd**; ETP 10d-MA at the 9.9th; SPX single-day at the 1.3rd of full history | **the registry contains ZERO put/call cells.** 4,984 days since 2006-11 and no pitch has ever used them. -> **C1, C2**. |
| gold/silver ratio | 6.869, the 56.3rd percentile of the trailing year and the 32.4th of full history | not an extreme. Dismissed on the number. |

---

## 3. Live seasonal and cycle cells

- **Midterm year (year%4==2).** A conditioner on everything above, never an
  idea of its own. The seasonal board's read: book win 56.4% against 64.9%
  all-years over 1,099 midterm trades, so de-risk. The registry holds six
  independent reproductions of a midterm inversion on the Jackson Hole
  anchor. Every candidate below owes a midterm split.
- **August, last four sessions.** August ME-5 is 5-of-11 at -0.510% since
  2015 against 13-for-13 at +1.271% through 2014, and the pre-2014 versus
  2014+ month-profile Spearman is -0.39. This is why watchlist #12 is parked
  and why C9 is framed as the September turn rather than as an August
  month-end entry.
- **September.** The one month-of-year cell the registry does not close in
  the equity direction; what it closes is September post-opex VOL, which is
  the event sleeve's T3. -> C9.
- **November TLT** (watchlist #10) parks to a date, trading days 4-12 of
  November. Not live.
- Cycle-year x tape-state: the fragility dial at 88.9 in a midterm August is
  itself a cell, and it is C8.

---

## 4. Watchlist: all 27 active entries, verdict each

State computed in `00_watchlist_state.py` against the 2026-08-25 close.

| # | entry | today | verdict |
|---|---|---|---|
| 0 | TLT from the NFP close, long end at its 52w floor | still midterm; NFP is +7 td | **PASS**, arms 2027-01 |
| 1 | LQD against HYG at joint 52w extremes | HYG 0.00% off its high, LQD +1.36% above its low: state live | **PASS** on the trigger, which is episode COUNT (4 declustered since 2007 against the 8 required); a live state cannot move it |
| 2 | SVXY overnight into CPI | CPI is +11 td | **PASS**, out of horizon |
| 3 | GLD on a miner-led thrust | GDX rank5 99.2 and GLD rank5 96.8 both fire; GLD is **-13.68%** off its 52w high against the >-10% rung | **PASS** on the fourth condition, third session running |
| 4 | XLE on a crude pop in the [5%,6%) band | USO's one-day move was **-4.58%**, the wrong sign entirely | **PASS** |
| 5 | TLT with the IG complex pinned at 52w lows | TLT **+2.61%** above its low against the <=0.5% rung; IEF +1.45% and LQD +1.36% also outside | **PASS**, further away than yesterday after the duration rally |
| 6 | SPY on a skew spike alone | SKEW rank5 **47.2** against >=95; the midterm block stands | **PASS** |
| 7 | Fade a crude thrust with a print inside the hold | USO rank5 22.6 against >=90 | **PASS** |
| 8 | IHI at a 21d rank of 100 | IHI rank21 **96.4**; the family-wise p 0.933 blocker is unchanged | **PASS** |
| 9 | FXI break inside an intact thrust | FXI rank5 **68.7** against <=20 | **PASS** |
| 10 | TLT November month-position | parks to November | **PASS** |
| 11 | Short SPY at a 52w high with TLT at a 52w low | SPY -1.54% off its high against <=0.5%; TLT +2.61% above its low against <=1% | **PASS**, both legs |
| 12 | TLT into the month-end close | today is **ME-3**, inside the flat ME-3..ME-7 band, but the entry's second condition is literally "NOT August" and its first (ME-1 to ME-0 hit rate back above base) is unmoved | **PASS** |
| 13 | SPY on a vol pop inside a calm tape | VIX rank21 15.5 clears calm, but the day's VIX move was **-2.52%** against a >=+5% pop | **PASS** |
| 14 | Gold on an unconfirmed rate rise | the dollar leg fires hard (DX rank21 0.8); the yield leg is **-0.002pt** over 21 sessions against a +0.20pt floor | **PASS** on the rate leg |
| 15 | Long tech against healthcare after a rotation gap | one-day XLV-XLK gap **-0.60pp** against >=+3.0pp | **PASS** |
| 16 | Short the dollar on a rate rise it does not confirm | TNX rank21 **37.3** against >=65 | **PASS**; the one-sided state (dollar stretched, rates absent) is exactly what C3 tests as a bare parent |
| 17 | Crude through Jackson Hole at JH-6 | the anchor was 2026-08-20, now JH-2; XLE -2.65% off its high, inside the forbidden 5% band | **PASS** |
| 18 | Short TLT after a big up day from the 52w low zone | TLT's one-day move was **+1.10%** against the >=+1.5% rung, and it is +2.61% above the low against "within 4%" | **PASS**, closest it has been |
| 19 | Short KRE against XLF on a breadth washout | KRE rank5 **9.1**, state live; the trigger is a cost threshold on ex-crisis history that a new episode cannot move | **PASS**. C7 deliberately tests the OPPOSITE side so it is not this entry re-pitched |
| 20 | HYG across Jackson Hole at JH-5 | the anchor is gone; closed on credit | **PASS** |
| 21 | IEF against 0.52 TLT with ^TNX at a 52w high | ^TNX at **97.77%** of its trailing-252 high against the 99.75% rung, having moved AWAY as the yield fell 1.42% in five sessions | **PASS** |
| 22 | Narrow energy thrust cluster, two or three names at z10 >= 2 | the count is **0 of 11** | **PASS** |
| 23 | Cross-sectional new-high breadth with the index off its high | SPY -1.54% against the >2.0% leg, and raw-21d fragility 69.4 against the <=50 leg | **PASS**, both legs |
| 24 | OIH outright at a services-versus-E&P extreme | the spread is at the **2.8th** PIT percentile, just outside its 2.5 rung, and the trigger is a record four wins away | **PASS** |
| 25 | Sector washout into a 52w high, family form | the closest is XLI at rank5 6.0 with -4.35% off its high; nothing sits at rank5 <= 5 | **PASS** |
| 26 | XLU washout with the long end hit alongside | XLU rank21 **3.2** fires, but TLT rank21 57.5 and rank5 96.8 -- the long end RALLIED, so today is the version the entry itself measures at +0.090% against XLU's +0.132% base | **PASS**; utilities are dead in six expressions and are not re-opened |

No entry fires today. Nothing expires. No cheap re-run was available, which
is part of why the morning is spending its budget on virgin surfaces.

---

## 5. Candidates selected (12)

Asset classes touched: us_large, us_small, rates, credit, gold/miners,
metals, dollar/FX, international, sectors, volatility, positioning = **11**
(floor 4). Novelty axes: flow_mechanics, inversion, instrument_translation,
interaction_cell, relative_value, historical_analogue, event_fingerprint =
**7** (floor 4). Event-anchored: C9, C12. Price-state-anchored: C3, C5, C6,
C7, C10, C11.

| id | candidate | axis | class | why it is not a known corpse |
|---|---|---|---|---|
| C1 | Index put/call 10d-MA at a trailing-252d percentile <= 10 while equity P/C sits mid-range. Vehicle SPY, direction taken from the data. | flow_mechanics | positioning x us_large | the registry has zero put/call cells; the book reads EQUITY P/C only, and only as a sizing conditioner |
| C2 | ETP (ETF-options) put/call at a trailing-252d low, the hedging-demand gauge, on IWM | flow_mechanics | positioning x us_small | a different series and a different vehicle from C1; also virgin |
| C3 | The BARE dollar 21d PIT washout (rank <= 2), no rate leg. Vehicle UUP and DX-Y.NYB. | inversion | dollar_fx | it is the untested PARENT of watchlist #14 and #16, both parked on the missing second leg |
| C4 | The same dollar washout translated to the funding trade: long EEM | instrument_translation | international | the mechanism is the dollar rather than the country, so it sits outside the closed country-decoupling family |
| C5 | Metals-complex thrust BREADTH: 4 of 6 members at a 21d rank >= 95 at once | interaction_cell | gold/miners x metals | the single-name forms are closed (GDX fade, copper thrust, silver against gold); a breadth COUNT is a different object, and must clear the 2026-08-24 energy-count method trap head on |
| C6 | HYG printing a fresh 52-week high while SPY is at least 1% off its own. Vehicle SPY. | interaction_cell | credit x us_large | around 49 declustered episodes since 2007; the closed credit cells are LQD-versus-HYG quality and HYG-across-JH, neither of which is credit leading equity |
| C7 | Regionals washed out (KRE rank5 9.1) while big banks print a 63d rank of 96.8 at a 52w high: long KRE against short XLF | relative_value | sectors/financials | watchlist #19 is the SHORT-KRE side on a breadth washout; this is the opposite sign on a different trigger |
| C8 | The fragility dial's 10d-MA 63d at or above 85, as a directional analogue for SPY | historical_analogue | us_large | must confront the book's own documented negative (the dial-conditioned book-wide throttle, PIT t=-0.23) before anything else; a directional index trade is a different object from sizing the book, and small N is expected |
| C9 | The turn INTO September, entered at ME-3, on IWM | event_fingerprint | us_small | the closed month-end work is month-POSITION; this is month-of-year |
| C10 | XLRE snapping up (z10 +1.25) out of a 21d rank of 23.8 while the long end rallies | interaction_cell | sectors x rates | XLRE has zero registry mentions and the rate-sensitivity link is the stated mechanism |
| C11 | MOVE/VIX at the 81.7th percentile of the trailing year: bond vol rich to equity vol. Vehicle TLT or IEF. | interaction_cell | rates x volatility | the closed vol-carry cells are all VIX term structure; cross-asset vol relative value is unexamined |
| C12 | The run into NFP entered seven sessions early, on a four-vehicle grid | event_fingerprint | us_large x rates | the registry's dismissal was "at +8 td it is at the horizon cap"; today it is inside for the first time. The placebo ladder is the FIRST test, not the last, because the ladder is 10-for-10 in this registry |

Negative-registry collisions were checked for every candidate; the
differences are stated in the last column and each checker receives the
adjacent registry entries verbatim.
