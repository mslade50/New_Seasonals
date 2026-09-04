# Surface map — 2026-08-31 (Monday, ME-0 for August, midterm year)

State: `data/pitch_state.json` (generated 09:14), freshest bar **2026-08-28**.
Prior session is Friday 2026-08-28, so the cache is current. One stale tape
name (LEG), irrelevant to everything below.

Regime, stated once because it conditions every dismissal here:
SPY 769.35, **-1.10% off its 52w high**, +8.66% above the 200d, r21 81.0.
Fragility **ma10(63d) = 87.6** (raw 63d 89.0, raw 21d 68.3) — the exposure leg
is killed (Rule 1) and the dial sits at a level the registry has repeatedly
flagged as **out of sample for almost every cell measured in this repo**
(2026-08-26: credit cell max historical dial 68.0; dial >= 85 analogue is one
2021-12 episode). P/C fear **off** (44th pctile). Signals on: Defensive
Leadership, Low Absorption Ratio, VIX Range Compression.

**Today is ME-0, the last trading day of August.** Tomorrow opens September.

---

## 1. Calendar events x asset class

`calendar.events` inside [-5, +15] td. Eight entries, six distinct anchors.
The honest summary first: the registry has filed a calendar finding on **five
consecutive sessions** (08-24 .. 08-28) saying the late-August midterm anchor
inventory is exhausted in the strong sense, and nothing has moved since Friday
except that the month turn arrived.

| anchor | date | td | status |
|---|---|---|---|
| jackson_hole | 08-28 | **-1** | CLOSED on eight classes pre-speech, ten post-speech (08-27, 08-28). Post-JH duration pulse is midterm-inverted for the seventh time. Do not reopen. |
| nfp | 09-04 | +4 | CLOSED. Ladder is a plateau on four vehicles; September prints pay -0.038%, September-in-a-midterm **-0.676% on 3-3** (08-26 c12b). |
| ppi | 09-10 | +7 | CLOSED. Print session is one session wide and 2013+ only; parked to the PPI watch, which arms only on the eve. |
| cpi | 09-11 | +8 | CLOSED as a standalone and as a PPI/CPI containment object (08-27 a3_c3). |
| fomc_decision + vix_expiry | 09-16 | +11 | **Beyond the 10 td horizon cap.** Enters ~09-02. Spoken for by event sleeve T1/T2; T2's own gate (SPY r21 < 50) reads 91.3, so even the sleeve is off. |
| opex + quad_witching | 09-18 | +13 | Beyond the cap. September post-opex vol is registry-dead (the crush inverts). |

**Month turn (ME-0, today).** Not in `calendar.events` but it is the live
anchor. Registry status: the month-end anchor is **closed on five classes** —
equities (08-24), rates (the ME-9 parent, 08-19/08-24), FX (08-26), commodities
and metals (08-27), sector-conditioned (08-28). The turn INTO September at ME-3
is closed on a scanned-session charge and a max-of-12 permutation P of 0.238.
What is NOT closed, and is the reason a month-turn candidate is on the list at
all, is that **every one of those five closures measured close-to-close daily
returns.** The ME-0 close to ME+1 open **overnight** return has never been
measured in this repo on any anchor, and it is the only return the
auction-flow mechanism actually claims. See C1.

### Event x class grid

Rows are the six anchors; the four dead ones are dismissed once for all ten
classes rather than sixty times, because the kill is the anchor and not the
vehicle.

- **jackson_hole (-1 td)**: dismissed on all ten classes. Swept to exhaustion
  on 2026-08-27 (volatility, the eighth class) and 2026-08-28 (post-speech, ten
  classes, 210 cells, best cell permutation P 0.065). Eight independent midterm
  inversions.
- **nfp (+4 td)**: dismissed on all ten. The 4x6 pre-declared grid returned a
  rotation null P 0.338; September-midterm is -0.676%. The one class worth a
  sentence is **rates**, because the NFP/TLT-at-the-floor watch parks to the
  first NON-midterm print (2027-01) and 2026 is midterm — so rates is blocked by
  cycle rather than by evidence. Not examined further today.
- **ppi (+7) / cpi (+8)**: dismissed on all ten. Both came inside the horizon on
  08-28 and both closed the day they arrived. The containment form (a hold
  spanning both) is 92.3% redundant with "CPI in the hold".
- **fomc / vix_expiry / opex / quad (+11..+13)**: outside the 10 td horizon cap.
  Not examinable today; FOMC becomes reachable ~09-02.
- **month turn (ME-0, today)**: the only live anchor. Cross:
  - us_large — **C1** (overnight, a new object) and dismissal of every
    close-to-close form.
  - us_small — **C1** carries IWM as its second vehicle; the ME-3 to ME-2
    small-cap session has already passed this month (08-26 to 08-27) and is
    blocked on a scan charge regardless.
  - rates — the ME-9 entry window closed on 08-18; today is its EXIT date, not
    an entry. Dismissed as unactionable.
  - commodities, metals, FX — closed 08-26 / 08-27 on the equivalent windows.
  - credit, volatility, international, sectors — the sector-conditioned form
    closed 08-28; the other three have no month-turn literature here and are
    covered by the equity closure, since the mechanism (index rebalancing) is
    an equity and bond flow. Dismissed.
- **earnings cluster (not a macro anchor, but live)**: MDT 09-01, **AVGO
  09-02**, CPB 09-03, ORCL 09-08, ADBE 09-10, KR 09-11. The earnings calendar
  (`data/earnings_calendar.parquet`, 946 tickers, forward dates) has **never
  been used as an anchor by a pitch check** in this product's life. AVGO is two
  sessions out and sits at a **63d rank of 2.4** — a lagging mega-cap into its
  own print. That is a genuinely unswept surface. **C3.**

## 2. Tape extremes by class

Sorted over the whole 218-name tape (`scratch/_tape_sort_0831.py`,
`scratch/_tape_macro_0831.py`), not looked up by name.

| class | the extreme | verdict |
|---|---|---|
| us_large | SPY -1.10% off its high, r21 81.0, r63 19.0; CRM r5 99.6 / r21 100 / +22.4% in five sessions | The round-trip-breakout cell (near-high with a bottom-quartile r63) was tested and closed on 08-28. CRM is a single-name thrust: the 205-name washout reference class (08-27) closed the down side and the 28-name leadership class (08-28) closed the up side; a 1x single-name thrust fade is the IHI corpse. **Dismissed.** |
| us_small | IWM r5 20.2, z10 -1.07, -3.06% off its high while SPY is -1.10% | The relative laggard into the month turn. Carried as C1's second vehicle and as **C12**. |
| rates | **^TNX -0.53% from its 252d high (4.720 against 4.745)**, ^MOVE at the **44.4th percentile** of its trailing year and -7.94% over 21d | An orderly grind to a cycle yield high with bond vol compressed. This is not the MOVE/VIX ratio cell killed 08-26, which died because MOVE was a cheap denominator and the object there was the ratio. **C2.** |
| rates, cont. | TLT +1.88% off its 252d low, IEF +0.73%, LQD +0.88% | The tight IG rung needs TLT <= 0.5%; today **1.88%, PASS**. The SPY-high / TLT-low pair needs SPY within 0.5% of its high (-1.10%) and TLT within 1% of its low (1.88%) — **PASS on both legs**. The curve trade needs ^TNX within 0.25% of its 252d high; today **-0.527%, PASS**, and it was cost-blocked anyway. The TLT-bounce short needs a +1.5% day; Friday was -0.30%, **PASS**. |
| credit | **HYG -0.23% from its 52w high** while IG sits at 52w lows | The pure-repricing rung (IEF <= 1.5, LQD <= 1.5, HYG within 0.25 of its high) **fires today** and is blocked on an episode count of one, the live one. The 08-26 kill of the HYG-high cell was **depth-conditional**: worth +0.615pp when the index is at least 2% below its high, -0.042pp in the 1-2% band, and SPY is -1.10% today. **IWM is -3.06%, inside the live band.** That is the one honest way this cell is still open. **C4.** |
| gold_miners | GDX +29.79% over 21d, NEM +33.65%, both flushed Friday (-3.90%, -3.26%); GLD -3.24% on 2.86x volume | Fingerprint `749b2073856902b3` (long GDX, MOC, 5d) was pitched **2026-08-27** and is inside the 10 td repetition block. That idea was NOT approved, so there is no live position, but a repeat needs `changed_since`, and the registry's own note says violent miner thrusts above +10% a week are the wrong half. The *spread* form — the miners' 21d outperformance of the metal at +21.4pp — is a different object. **C18.** |
| metals | SLV -4.38% Friday, -12.16% over 63d, -7.66% below its 200d, while FCX z10 +1.60 and XME r21 85.3 | Precious flushed, industrial thrusting. Silver's drawdown-thrust conditioner is registry-dead (a U-shaped noise carve, 08-10, confirmed on country equity 08-28). The industrial leg is **C10**. |
| energy | **OIH minus XOP 63d spread -13.93pp, PIT trailing-252 pctile 4.0**; SLB +17.2% over 21d; VLO at a fresh 52w high, +44.5% over 63d | The OIH watch turns on at <= 2.5 pctile and today reads **4.0 — CHECK, near-armed on the loosened rung. C6.** VLO is the leadership shape closed on 28 names (08-28); dismissed. The narrow-thrust count needs 2-3 energy names at z10 >= 2.0; today the count is **0** (XLE -0.10, XOP +0.23, USO -0.03, OIH -0.52 on `pitch_lab.zscore`). **PASS.** Crude through Jackson Hole at JH-6 — the anchor has passed, **PASS**. The crude-pop cells need a one-day USO move in [5%,6%) at 1.50 ATR; Friday was -0.24%, **PASS**. |
| dollar_fx | DX-Y.NYB -1.88% from its 52w high, r5 87.7, +0.91% over 5d; UUP -1.47% | The dollar is **confirming** the rate rise, which is the inversion of the parked "short the dollar on an unconfirmed rise" and of the bare washout (midterm-parked). The washout rung is a 21d rank <= 2; today DX reads 42.9. **PASS.** The confirmed form is untested. **C14.** |
| international | EEM r63 2.0 with r21 65.1; FXI r21 34.1; EWZ z10 +1.21 | The China break needs FXI r5 <= 20 with r21 >= 80; today 36.1 and 34.1. **PASS.** The country-decoupling family and the thrust-in-drawdown inversion are both closed (08-28, 11-name family, common excess -0.230%). **Dismissed.** |
| volatility | **^VIX3M 17.48 = the 0.4th percentile of its trailing year and its exact 52w low**; ^VIX 14.43 (3.2nd pctile); **^SKEW 149.77, r21 93.3, +7.06% over 21d**; VIX3M/VIX 1.211 at the 83.3rd pctile | The VIX3M level floor was tested and closed **four sessions ago** (08-27 c4: the threshold ladder inverts and the cell underperforms its own local control). Term-structure percentile closed both directions (08-13). The skew-alone watch needs `pct_rank(^SKEW,5) >= 95`; today **83.3, PASS**, and it is midterm-blocked besides. What is left is the **ratio of tail premium to at-the-money vol**, SKEW rich while VIX3M is at a floor — genuinely adjacent to two corpses, and worth exactly one checker slot with that stated up front. **C5.** |
| sectors | XLU r21 9.5 (the utilities watch needs <= 5, **PASS**), IYR 12.3, VNQ 11.9, XLRE 15.5 — the whole duration-equity complex at rank floors with ^TNX at a 52w yield high; SMH **r63 0.8** inside a +86% year; staples split GIS r63 100 and CAG 98.4 against HRL -14.98% and TSN -8.66% | Utilities are dead in **six** expressions and real estate is closed with nothing to park (08-26). The semis-laggard watches need r21 >= 90 with r63 <= 10; **zero holders on the tape today**, and SMH reads r21 40.1 / r63 0.8 / r5 28.2. **PASS on both.** The staples dispersion is real but has no vehicle better than XLP, whose own r21 is 42.1; dismissed on vehicle. |

## 3. Seasonal and cycle cells

- `seasonality.board_candidates` carries **0 A/B-grade setups** and four
  midterm de-risk context rows (book win 56.4% against 64.9%, OVS 55.4%
  against 67.6%, LT Trend ST OS 53.7% against 67.6%, Indices Oversold Bounce
  59.0% against 64.5%). Context, not a trade.
- The board's put/call row is stale (reading as of 2026-08-04) and the live
  state contradicts it: P/C fear is **off** at the 44th percentile, not the
  4th. Not used.
- **Midterm is now a REQUIRED round-2 test**, not an optional one (08-26 method
  trap: the inversion killed or blocked five unrelated candidates in a single
  morning, and the Jackson Hole lane recorded eight independent inversions).
  Every candidate below carries a midterm split in round 2.
- Month-of-year: TLT's own 10td lag-1 forward return runs **Sep -0.220%**,
  second-worst of twelve, against Nov +1.059%. That is the control any
  September rates idea owes, and it is why **C8** is framed short rather than
  long.

## 4. Watchlist verdicts (34 active, every entry)

Armed and near-armed first, then the passes with today's number.

- **[24] OIH at a services-versus-E&P 63d extreme — CHECK.** PIT pctile **4.0**
  today against a 2.5 arm. Not armed on the literal rung; carried as C6 so the
  loosened rung is priced honestly rather than assumed.
- **[31] Pure rates repricing, IG at 52w lows with HY at a high — rung FIRES,
  blocked.** IEF +0.73%, LQD +0.88%, HYG -0.23%, all inside tolerance. The arm
  condition is >= 8 declustered episodes; there is still exactly **one**, and it
  is the live one. Its sibling depth question is what C4 tests instead.
- **[26] Utilities washout with the long end hit — PASS.** XLU r21 **9.5**
  against a <= 5 arm, and TLT is r21 64.7 rather than hit. Utilities are dead in
  six expressions regardless.
- **[5] TLT with the IG complex pinned — PASS.** TLT is **+1.88%** off its low
  against a <= 0.5% arm; IEF and LQD clear.
- **[11] Short SPY at a 52w high with TLT at a 52w low — PASS.** SPY **-1.10%**
  (needs >= -0.5%), TLT **+1.88%** (needs <= 1.0%). Both legs off.
- **[21] Duration-neutral IEF against TLT at a 52w yield high — PASS.** ^TNX
  **-0.527%** from its 252d high against a 0.25% arm; cost-blocked anyway.
- **[18] Short TLT after a big up day from the 52w low zone — PASS.** TLT's
  one-day return was **-0.30%**, not >= +1.5%.
- **[6] SPY on a skew spike alone — PASS twice.** `pct_rank(^SKEW,5)` is
  **83.3** against a >= 95 arm, and the entry is cycle-blocked in a midterm
  year.
- **[22] Narrow energy thrust cluster — PASS.** The count of members at
  z10 >= 2.0 is **0**; the arm is 2-3.
- **[30] Semis at a 63d rank floor and [33] the pooled laggard still falling —
  PASS.** SMH reads r21 **40.1**, and the joint state (r21 >= 90, r63 <= 10) has
  **zero holders** across the whole tape today.
- **[27] Bare dollar washout — PASS.** DX r21 is **42.9** against a <= 2 arm,
  and it is midterm-parked to 2027 besides.
- **[9] FXI break inside an intact thrust — PASS.** FXI r5 **36.1** (needs
  <= 20), r21 **34.1** (needs >= 80).
- **[4] and [7] XLE and the crude-thrust fade — PASS.** USO's last session was
  **-0.24%**; both need a 5%-plus one-day pop.
- **[3] GLD on a miner-led thrust — PASS on the fourth condition.** GDX r5 is
  25.8, below the >= 95 arm, and GLD is **-17.55%** off its 52w high against the
  added "within 10%" condition.
- **[17] Crude through Jackson Hole at JH-6 — PASS, anchor gone.** The
  conference was 2026-08-28.
- **[32] IEF one session out of the Jackson Hole close — PASS, cycle-blocked**
  to 2027-08-27 (midterm).
- **[0] NFP with TLT at the floor — PASS, cycle-blocked** to the first
  non-midterm print, 2027-01.
- **[10] TLT on the November month-position effect — PASS, date-parked** to
  November.
- **[12] TLT into the month-end close from ME-9 — PASS, window gone.** The entry
  session was 2026-08-18; today is the **exit** date, not an entry.
- **[2] SVXY overnight into CPI — PASS.** Arms on the CPI eve (09-10), and on a
  leave-one-year-out floor of 40-50 bps against 19.7 today.
- **[1] LQD against HYG at joint extremes — PASS**, episode count still 4.
- **[8] IHI thrust — PASS.** IHI r21 87.7 against a rank-100 arm, and it is
  reference-class dead.
- **[13] SPY on a vol pop inside calm tape — PASS.** VIX fell **-0.55%** on
  Friday; the arm needs +5%.
- **[14] Gold on an unconfirmed rate rise — PASS.** DX r21 42.9 against a <= 15
  arm.
- **[15] Tech against healthcare on a rotation gap — PASS.** Friday's XLV minus
  XLK gap was **+1.31pp** against a >= 3.0pp arm.
- **[16] Short the dollar on an unconfirmed rate rise — PASS**, and today is its
  inversion, since the dollar IS confirming. Carried forward as C14.
- **[19] Short KRE against XLF — PASS**, and financials are closed in both
  directions (08-26).
- **[20] HYG across the Jackson Hole speech — PASS**, anchor gone.
- **[23] Cross-sectional new-high breadth — PASS.** Needs SPY more than 2.0%
  below its high; today **-1.10%**.
- **[25] Sector washout into a 52w high, family form — PASS.** No SPDR is at a
  5d rank <= 5 while within 5% of its high; the closest, XLV, is r5 11.9 and
  -2.57% off.
- **[28] HY at a fresh 52w high while the index has not — PASS on depth.** SPY
  is **-1.10%**, inside the dead 1-2% band the 08-26 kill identified. This is
  precisely the number C4 re-asks with IWM (-3.06%) in the index slot.
- **[29] The ME-3 to ME-2 small-cap session — PASS, window gone.** That session
  was 2026-08-26 to 08-27, and it is blocked on a scan charge.

Nothing on the watchlist is armed. Two are near-armed (24 at 4.0 against 2.5;
31's rung fires but its episode count does not move).

## 5. Axis and grade read (scoreboard)

`scoreboard.lifetime` has **4 graded ideas** — B avg +0.448R on 3, C +0.146R on
1, hit rate 100% on a sample that cannot possibly support that. By axis:
event_fingerprint +0.622R on 2, interaction_cell +0.146R on 1, relative_value
+0.099R on 1, inversion ungraded. **The graded count is a handful and no axis
read is warranted yet**, which is what this section is supposed to say when that
is true. No axis is being up- or down-weighted today.

## 6. Candidates selected (12)

Classes touched: us_large, us_small, rates, credit, volatility, energy, metals,
gold, dollar_fx, sectors — **ten**. Axes: flow_mechanics, interaction_cell,
relative_value, inversion, event_fingerprint, instrument_translation — **six**.
Event-anchored: C1, C3, C8. Price-state anchored: C2, C4, C5, C6, C10, C12,
C14, C18.

| id | candidate | axis | class |
|---|---|---|---|
| C1 | The month-end closing auction's **overnight** reversal: SPY and IWM MOC on ME-0 to MOO on ME+1. Never measured; every prior month-end closure is close-to-close. | flow_mechanics | us_large, us_small |
| C2 | Yield at a 52-week high with **bond vol compressed** (^MOVE at its 44th level percentile, -7.94% over 21d): the orderly grind, traded on TLT and IEF. | interaction_cell | rates |
| C3 | Pre-print drift in a **deeply lagging mega-cap**: AVGO at a 63d rank of 2.4 reporting in 2 td. First use of the earnings calendar as a pitch anchor. | flow_mechanics | us_large, sectors |
| C4 | HY at a 52-week high while the **small-cap** index sits 3.06% below its own — the 08-26 depth kill re-asked with an index that is inside the live band. | interaction_cell | credit, us_small |
| C5 | Tail premium rich against at-the-money vol at a floor (SKEW against VIX3M), as a **ratio** rather than the closed rank conjunction. | inversion | volatility |
| C6 | OIH outright at the services-versus-E&P 63d spread extreme, PIT pctile 4.0 on a 2.5 arm — the loosened rung, priced honestly. | relative_value | energy |
| C8 | Short duration **into September** with the 10-year at a 52-week high: TLT's second-worst month crossed with the live price state. | interaction_cell | rates |
| C10 | Industrial metals thrusting while precious flushes: FCX and XME against SLV and GLD. | relative_value | metals |
| C12 | The small-cap laggard into the month turn: IWM against SPY, r5 20.2 against 52.4. | relative_value | us_small, us_large |
| C14 | The dollar **confirming** a rate rise, the untested inversion of the parked unconfirmed form. | inversion | dollar_fx |
| C15 | Intraday shape of the ME-0 session: does the last hour of the month-end close carry the flow, and is it given back at the next open? Round-2 support for C1 on the 15-minute cache, a data surface no pitch check has used. | flow_mechanics | us_large |
| C18 | The miners' 21-day outperformance of the metal at +21.4pp: the GDX-against-GLD spread as a fade, distinct from the blocked long-GDX fingerprint. | relative_value | gold_miners |

---

## 7. Addendum — the 5:10 AM run (appended 09:35)

The scheduled task ran at 05:10, wrote real work into this folder, and died
before publishing. There is no journal record for 2026-08-31, so the morning
was never delivered. Its candidates are part of today's sweep and are recorded
here rather than repeated: the surviving question from it is carried forward as
C19 and handed to the metals checker.

| id | candidate | class | status |
|---|---|---|---|
| E1 | Month-end rebalance OVERNIGHT, ME-0 close to ME+1 open, on SPY/QQQ/IWM/DIA/TLT | us_large, us_small, rates | Same object as C1; independently re-checked this hour. Its own numbers: pooled 4-vehicle equity excess over the unconditional overnight +13.5 bps at t 3.36, but **the August turn alone is -0.053% at a 53.8% hit (N=26)** and today's exact cell (August turn, midterm year) is **-0.057% on N=6 at a 50.0% hit**. Cost: SPY excess +7.55 bps against 20 bps needed for 5x. |
| E2 | The whole-metals-complex break read as a LONG DOLLAR signal | dollar_fx | Reference class over six complex-break families: Cochran Q p 0.637, I-squared 0.0%, common excess +0.053%; horizon multiplicity max-of-10 p 0.264. |
| E3 | The same break traded as a SHORT on the complex | metals, gold_miners | Per-leg: short GLD **-2.6x cost**, short GDX **-1.1x cost**, both with top-2 concentration at 202% and -233% of total. **The SLV leg is the exception and was never adjudicated** — carried forward as **C19**. |
| E4 | Two-member idiosyncratic shock inside an already-washed-out XLU | sectors | Utilities are dead in six expressions; the check was written to test whether member-level shock adds to the closed ETF-level cell. |
| E5 | Equipment and analog semis at simultaneous 63d rank floors while NVDA runs | sectors | The member-breadth form of the two dead semis-laggard watch entries. |
| E6 | One-day ^SKEW rise >= 3% on a session the ^VIX FELL, as a CO-MOVEMENT trigger | volatility | Adjacent to C5 and to two closed level/rank forms. |
| E7 | September month-of-year on the DOLLAR, entered at the August month-end close | dollar_fx | **Max-of-12 permutation P = 0.391 at its best horizon** (h=3, mean +0.347%); turn-on needs a September mean of +0.532%, 1.5x the observed. h=1 and h=5 return P 1.000 and 0.999. |
| E8 | Pooled: a single name at a 21d rank <= 2 while its own sector ETF is at r21 >= 75 | us_large, sectors | The pooled parent of the TJX washout, which is reference-class dead on 205 names. |
| E9 | Buy the month's largest 21-day winner at the month-end close, fixed 18-name cross-asset basket | cross-asset | ME-0 h=2 +0.347% (t 2.14) against a non-ME +0.087%, but h=3 and h=5 collapse to t 1.37 and 1.55, and **23.6% of the month-end picks are metals** (GDX 38, SLV 30, GLD 7), so the basket is a metals-momentum trade wearing a rebalance label. |
| E10 | QQQ at a 63d rank extreme while SPY sits near its 52w high | us_large | Live (QQQ r63 8.7, SPY -1.10% off). |
| E11 | Fragility dial >= 85 with a calm VIX, traded on VOLATILITY (short SVXY) rather than direction | volatility | Deliberately not the directional dial claim closed 2026-08-27. |

**C19 — SHORT SILVER after a whole-metals-complex break.** Trigger: GLD, SLV
and GDX each close <= -2% on the same session. It **fired Friday 2026-08-28**
(-3.24% / -4.38% / -3.90%), so the entry is today. Round-1 numbers from the
earlier run, to be reproduced or refuted rather than banked: episode mean
**+0.350% at h=5 = 5.8x a 6 bp round trip**, +0.772% at h=3, +0.531% at h=1
(t 1.976, 56.3% hit); edge over the all-days control +0.618pp at h=5;
**top-2 episodes are 3% of a +41.66pp total**; decluster-stable at gap
5/10/21 (+0.350 / +0.313 / +0.507); era-stable (pre-2018 +0.380, 2018+
+0.278). The six-family reference class puts metals/SLV at a permutation
max-of-6 **p 0.0130 at h=1 and 0.0287 at h=3** — a CONFIRMING class, which is
the rarer outcome and gets scrutinised harder rather than banked. The open
objection is that h=5 is 60-59 at sign p 0.500, i.e. a mean effect from
asymmetry rather than a hit-rate effect.

**Morning totals after the addendum**: ~20 distinct candidates across ten
asset classes (us_large, us_small, rates, credit, volatility, energy, metals,
gold_miners, dollar_fx, sectors) and six novelty axes.
