# Surface map, 2026-08-21 (Friday, monthly OPEX day, midterm year)

Tape read: freshest bar 2026-08-20, pipeline 7/7 green, no state warnings.
Trading day 15 of 21 in August. Fragility ma10(63d) **89.1** (raw 21d 57.6,
raw 63d 86.7) with P/C fear OFF, so the book's six dip-buy carriers are
ZEROED today and the exposure leg is off. Signals on: Defensive Leadership,
Low Absorption Ratio.

Index state: SPY -1.96% from its 52w high on a -1.96% five-day slide (5d rank
9.5), QQQ -4.62% off (5d rank 10.3), IWM -2.43% off. VIX 16.01 after a +7.52%
session, 21d rank only 40.9. ^SKEW 143.23 at a 5d rank of 97.6 and a 63d rank
of 96.0.

## Scoreboard read before selecting

Four graded ideas lifetime (avg +0.372R, 4-for-4). By axis: event_fingerprint
2 at +0.622R, interaction_cell 1 at +0.146R, relative_value 1 at +0.099R. That
is a handful, not a signal. No axis is steered for or against today.

## 1. Calendar x asset class

Six events inside the [-5,+15] window. An idea entered MOC today with a legal
1-10 td horizon exits on or before **2026-09-04**, so PPI (+13) and CPI (+14)
are outside every tradeable hold and their twenty cells are dismissed on
horizon grounds alone, not on merit. `vix_expiry` (-2) is past, and the
2026-08-17 registry finding is that opex and VIX expiry are ONE anchor sharing
189 of their N days, so its ten cells fold into the opex row.

That leaves opex (0), Jackson Hole (+5) and NFP (+10) as live anchors: thirty
cells.

| class | opex (today) | Jackson Hole (+5) | NFP (+10) |
|---|---|---|---|
| US large | DEAD. The run out of August opex on SPY was closed 2026-08-20 (120-cell month x horizon x vehicle grid, rank 5 of 120; offset ladder self-contradicting) and the run into it died 2026-08-07. | DEAD 2026-08-18, the sweep that completed JH to five asset classes. | DEAD 2026-08-07, post-NFP equity direction swept and empty. |
| US small / breadth | DEAD, same 2026-08-20 closure. Noting the state HAS inverted (IWM is 2.43% off its high, and the kill's own conditioner pays +1.100% in that half) — dismissed anyway as a sign recovered from a corpse, the exact trap the 2026-08-07 registry entry names. | DEAD 2026-08-11, short IWM at JH-13 wrong-signed in midterms; the long side is the same corpse reversed. | Not examined: the 10-session run into NFP is the same run-into-the-event construction that died on opex 2026-08-07 and on JH 2026-08-13, and the one NFP x equity cell measured (+0.129% against a +0.221% control) is its parent. |
| rates | **CHECK (C4).** The opex anchor has only ever been tested on equity vehicles. TLT and IEF carry their own listed option complexes with the same third-Friday expiry, so the crossing is a real cell and it is unexamined. | DEAD 2026-08-13, long TLT across the ten sessions into JH. | PARKED, watchlist 0: midterm-dead (+0.071%, t=0.17) and alive outside midterms; turns on 2027-01. 2026 is a midterm year. |
| credit | **CHECK (C5, folded into the cross-asset opex script).** Same argument as rates, and HYG/LQD have never been crossed with any expiry anchor. | **CHECK (C9).** Credit is one of only two classes the Jackson Hole sweep never reached. | Not examined: NFP x credit inherits both the dead post-NFP direction cell and the 2026-08-11 finding that HYG into a macro print is a pre-2018 fossil (+17.8 bps to -2.8 bps). |
| gold / miners | **CHECK (C3).** GLD is one of the most heavily optioned ETFs listed and the opex anchor has never been run on it. It is also live in a maximal thrust, so the cell is not hypothetical. | DEAD 2026-08-13, long GLD across the run into JH. | Not examined: the 2026-08-07 sweep killed pre-event windows on the event's own instrument (GLD into CPI underperforms its own drift), and gold into NFP is the same construction one release later. |
| other metals | Not examined on any anchor. Silver's event behaviour is gold's with more beta, and the 2026-08-11 kill of a second metals leg beside a live one is the standing reason not to spend an event check here while a gold candidate is live. Its PRICE state is examined below (C8). | Not examined, same reason. | Not examined, same reason. |
| energy | **CHECK (C5, same script).** Unexamined crossing. | DEAD 2026-08-20 (long crude at JH-6), and the watchlist entry's own second leg blocks it today: 0 of 26 anchors ever combined XLE within 5% of its 52w high with USO's 63d rank under 40, which is exactly today, and the one anchor that came closest is the worst episode in the sample. The JH-6 anchor day was also yesterday. | Not examined; the energy x macro-print family is 0-for-4 (XLE washout into CPI, USO thrust fade, crude pop into a print, XLE at a high with crude at a floor). |
| dollar / FX | **CHECK (C5, same script).** Unexamined crossing. | DEAD 2026-08-13, long DX futures across the run into JH. | DEAD 2026-08-07, short DX and short UUP on the NFP trigger, both wrong-signed in midterms. |
| international | Not examined on opex: the anchor's mechanism is US listed-option flow and EFA/EEM trade a foreign session, so a US expiry cell would be measuring the SPY beta. Dismissed on mechanism, not on data. | **CHECK (C9, same script as credit).** The other class the JH sweep never reached. | Not examined; the US-flow argument does not apply here, but the post-NFP direction parent is dead and EM adds beta rather than a thesis. |
| volatility | The post-opex short-vol window IS the book's `V4_POSTOPEX_VOL` sleeve trade and it enters TODAY (long SVXY 1228 shares, exit 2026-08-26 MOC). Any short-vol idea here doubles a live sleeve position. The August carve-out question was settled 2026-08-20 in the sleeve's favour. DISMISSED on book overlap. | Not examined as a vol cell. The registry's post-event vol sweep (NFP, FOMC, VIX expiry) came back empty with opex the only survivor, and a JH long-vol idea would also collide with the live SVXY leg over 8/21-8/26. Dismissed. | Not examined; post-NFP vol is explicitly in the swept-and-empty list. |

## 2. Tape extremes by class

| class | the outliers, and the verdict |
|---|---|
| US large | SPY 5d rank 9.5 at -1.96% off its high; QQQ 5d rank 10.3, 63d rank 17.1. A shallow dip in a stretched tape. Plain dip-buying is not an idea (registry: SPY 5d <= -1% already pays +0.219% over N=511 unconditionally). Used as a CONDITIONER in C2 and C6 rather than as a candidate of its own. |
| US small / breadth | IWM 5d rank 13.1, XRT 5d rank 13.1 with a 63d rank of 66.7. Nothing separates small caps from the index move. Dismissed. |
| rates | TLT +1.22% above its 52w low, IEF +0.89%, LQD +0.69%, the whole IG complex pinned. Watchlist 5 wants TLT within 0.5% and it is at 1.22%. ^TNX 4.70, only 1.03% below its own 52w high. Dismissed as a price state; kept as the opex vehicle in C4. |
| credit | **HYG -0.29% from its 52-week HIGH while LQD sits +0.69% above its 52-week LOW.** Two live readings. (a) The joint-extreme pair is watchlist 1 and its trigger is episode count, still 4 declustered instances since 2007, unchanged today: PASS. (b) The genuinely new reading is credit AT its high while the index is in a five-day washout: **CHECK (C2)**. |
| gold / miners | **GDX +30.2% over 21 sessions at a 21d rank of 99.6, NEM +33.3%, GLD +9.5% at a 5d rank of 86.9.** Watchlist 3 fires today on all four of its conditions for the first time since it was parked: **CHECK (C1)**. |
| other metals | **SLV -41.61% from its 52w high yet +14.35% over 21 sessions and +6.02% over 5**, against GLD -16.26% off its high. A 25pp drawdown divergence inside a joint thrust: **CHECK (C8)**. FCX at a 52w high, +3.08% on the day: dismissed as single-name copper beta with no cell behind it. |
| energy | XLE, XOP, COP and EOG all AT 52-week highs (0.00% off) with XLE z10 1.97, while USO's 63d rank is 23.8. The producers-at-a-high-with-crude-at-a-floor cell died 2026-08-19 and the crude-pop band cell is watchlist 4, with today's +2.77% nowhere near its [5%,6%) band. PASS on both. |
| dollar / FX | UUP 21d rank 2.4, DX-Y.NYB 21d rank 3.2, the deepest dollar washout of any recent morning. Killed outright 2026-08-20 on concentration plus a backwards dose response (the bounce SHRINKS as the washout deepens, and today is the 91st percentile by shallowness). Watchlist 17's short side needs ^TNX 21d rank >= 65 and it is 50.0. PASS. |
| international | EWJ 5d rank 3.6 (-4.27%) while EFA holds at -1.23%; FXI 5d rank 80.6; EEM 63d rank 11.5; EWZ 21d rank 10.7. Japan is the one country never swept here: **CHECK (C11)**, with the standing family caveat that FXI, EWZ (twice) and KWEB all died to the same reference-class argument. |
| volatility | ^SKEW 143.23, 5d rank 97.6 and 63d rank 96.0, while VIX's own 21d rank is 40.9. The skew top pole with the index in a dip is watchlist 6's live half: **CHECK (C6)**. ^MOVE 73.18, 36% below its high, no cell (bond-vol level and one-day spikes both died 2026-08-18). |
| sectors / industry | **Banks 92.3% of a 13-name complex at a 5d rank <= 20 with a median 63d rank of 69.8** — the strongest breadth reading of any industry today, and the 63d median sits a whisker inside watchlist 10's trend-BROKEN line: **CHECK (C7)**, taking the untested LONG side. Semis 50% breadth with a median 63d rank of 8.7 (SMH 63d rank 1.2): the deep-laggard family is dead five ways (SMH/QQQ pair, semis outright, base breakout, laggard cross-section, semis into NVDA) and the breadth leg fails at 50%, so PASS. Healthcare and staples at 63d ranks of 98-100 with Defensive Leadership ON: sector-vs-index pairs on a leadership trigger are registry-dead and the index-level read IS the book's own fragility signal. PASS. |
| single names / earnings | **WMT -9.15% on the session it reported a +9.3% EPS beat and a revenue beat; ROST -6.5% over five days on a +36.4% beat; TJX z10 -2.49 on a beat.** Three simultaneous instances of beat-the-number-and-get-sold. The registry killed the PRE-print washout and the retail earnings CLUSTER anchor; the post-print beat-and-drop is a different event and is unexamined: **CHECK (C10)**. |

## 2b. What the book is actually staged in today

Recovered by fixing `scripts/build_pitch_state.py`, which was reading
`Ticker` / `Strategy_Name` off staging tabs whose columns are `Symbol` /
`Strategy_Ref`, so every staged row arrived with a null ticker and the
stage-C overlap check has been blind since the pitch state was built.

| tier | name | side | strategy | exit |
|---|---|---|---|---|
| liquid | LUV, UNH | LONG | Oversold Low Volume | 2026-09-04 |
| liquid | NEM | SHORT | Overbot Vol Spike | 2026-08-25 |
| overflow | CINF, MOD, POWI | LONG | Oversold Low Volume | 2026-09-04 |
| overflow | AGI, AU, CGAU | SHORT | Overbot Vol Spike | 2026-08-25 |
| overflow | APA, ARIS | SHORT | Overbot Vol Spike | 2026-08-25 |
| overflow | GL | LONG | LT Trend ST OS | 2026-08-24 |

The book is fading the gold-miner blow-off with **four short legs over
exactly 8/21 to 8/25**, which is material to C1 and C8 and is fed to the
checker. The event sleeve separately enters `V4_POSTOPEX_VOL` today, long
SVXY 1228 shares to 2026-08-26, which is why the whole volatility row above
is dismissed on book overlap.

## 3. Seasonal and cycle cells

Midterm year (2026, year%4==2) conditions everything above rather than
standing on its own. The seasonal board carries no A or B graded setups today;
its live content is four midterm de-risk regime rows (book win 56.4% vs 64.9%,
OVS 55.4% vs 67.6%, LT Trend ST OS, Indices Oversold Bounce) and a stale
2026-08-04 put/call complacency row that the live reading contradicts (the
board says 4th percentile, the live pc_fear state says 52nd). No seasonal cell
is a candidate. Mid-August midterm seasonality is registry-dead (N=6, carried
by 2002). Month-end is six sessions out: the equity turn-of-month sits in the
famous-calendar-cells-arbitraged-away entry, and the TLT month-end anchor is
watchlist 13, gated on TLT being more than 3% above its 52-week low when it is
at 1.22%. PASS.

## 4. Watchlist verdicts (21 active, every one accounted for)

| # | entry | today | verdict |
|---|---|---|---|
| 0 | TLT from the NFP close at the 52w rates floor | 2026 is a midterm year; turns on 2027-01 | PASS |
| 1 | LQD vs HYG at joint 52w extremes | the state IS live (HYG -0.29% off its high, LQD +0.69% off its low) but the trigger is episode count and it is still 4 | PASS |
| 2 | SVXY overnight into CPI | no CPI until 2026-09-11, and the LOYO floor is unchanged at 19.7 bps against a 40-50 bp trigger | PASS |
| 3 | GLD on a miner-led thrust the metal has not joined | **GDX 5d rank 96.4 >= 95, GLD 5d rank 86.9 < 95, no CPI or PPI inside a 1-10 td hold, no live GDX position (the last one exited 2026-08-17). All four conditions clear for the first time.** | **CHECK (C1)** |
| 4 | XLE on a crude one-day pop in the [5%,6%) band | USO's one-day move is +2.77%, outside the band and under the 1.50 ATR floor | PASS |
| 5 | TLT with the IG complex pinned at 52w lows | the tight rung needs TLT within 0.5% of its low; it is 1.22% | PASS |
| 6 | SPY on a skew spike alone | SKEW 5d rank 97.6 clears and the dip leg is live for the first time (SPY -1.96% off its high against the required >1%), but the second arming leg is non-midterm and 2026 is a midterm year | **CHECK (C6)** — one leg armed, one dead, and the entry itself warns the dip may be doing the work |
| 7 | crude thrust fade with a print inside the hold | USO 5d rank 83.7 (needs >= 90) and 63d rank 23.8 (needs <= 20) | PASS |
| 8 | IHI medical-device thrust | rank 98.8 not 100, and the reference-class blocker is unchanged | PASS |
| 9 | FXI five-day break inside an intact thrust | FXI 5d rank is 80.6, needs <= 20 | PASS |
| 10 | industry breadth washout with the trend BROKEN | banks are at 92.3% breadth with a median 63d rank of 69.8, just inside the broken half, and the untested leg it names is a <=4-name LONG selection rule against the alphabetical placebo | **CHECK (C7)** |
| 11 | TLT on the November month-position effect | parks to 2026-11-05 | PASS |
| 12 | short SPY at a 52w high with TLT at a 52w low | SPY is 1.96% off its high, needs 0.5% | PASS |
| 13 | TLT into the month-end close at ME-9 | the ME-9 anchor was 2026-08-18 and TLT is 1.22% above its 52w low against a >3% trigger | PASS |
| 14 | SPY on a vol pop inside a calm tape | VIX 21d rank 40.9, needs <= 25 | PASS |
| 15 | gold on an unconfirmed rate rise, both dials at force | the dollar leg clears at rank 3.2, the yield leg is a +0.04pt 21-session rise against a +0.20pt floor | PASS |
| 16 | tech against healthcare after a rotation gap | today's one-day XLV-minus-XLK gap is -1.58pp, wrong direction | PASS |
| 17 | short the dollar on an unconfirmed rate rise | ^TNX 21d rank 50.0, needs >= 65 | PASS |
| 18 | crude through Jackson Hole at JH-6 | the anchor day was yesterday, and the entry's own second leg forbids it with XLE at its 52w high | PASS |
| 19 | short TLT after a big up day from the 52w low zone | TLT's one-day move is -0.82%, needs >= +1.5% | PASS |
| 20 | short KRE against XLF on a bank-breadth washout | the breadth state is live at 92.3%, but the trigger is a cost threshold on history (+0.35% at h=3 ex-crisis) and one new episode does not move it | PASS, though the same state feeds C7 from the long side |

## 5. Selected candidates

| id | candidate | axis | class | anchor |
|---|---|---|---|---|
| C1 | Long GLD on a miner-led thrust the metal has not joined | interaction_cell | gold_miners | price-state |
| C2 | Long the index on a five-day washout while high yield sits at its 52-week high | interaction_cell | us_large x credit | price-state |
| C3 | The post-opex window on gold, a class the opex anchor has never been crossed with | event_fingerprint | gold_miners | event |
| C4 | The post-opex window on duration | event_fingerprint | rates | event |
| C5 | The post-opex window on the dollar, energy and credit, completing the cross-asset sweep | event_fingerprint | dollar_fx / energy / credit | event |
| C6 | Long the index on a skew top-pole spike inside a five-day dip | interaction_cell | volatility x us_large | price-state |
| C7 | Long the four most-washed banks on a 92% breadth washout with the 63-day trend broken | flow_mechanics | financials | price-state |
| C8 | Long silver against gold on the drawdown divergence inside a joint thrust | relative_value | other_metals | price-state |
| C9 | Jackson Hole at JH-5 on credit and international, the two classes the sweep never reached | event_fingerprint | credit / international | event |
| C10 | The beat that gets sold: a positive EPS surprise met with a large next-session decline | event_fingerprint | us_large single names | event |
| C11 | Long Japan on a five-day washout while developed international holds | inversion | international | price-state |

Eleven candidates, five axes (interaction_cell, event_fingerprint,
relative_value, flow_mechanics, inversion), nine asset classes, six
event-anchored and five price-state-anchored.
