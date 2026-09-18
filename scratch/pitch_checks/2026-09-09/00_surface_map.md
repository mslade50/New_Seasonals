# Surface map — 2026-09-09 (Wednesday, midterm year)

Bars through **2026-09-08**. One warning: `LEG` stale. Everything below is
computed in `00_recon.py` unless it cites the state file.

**This is the first morning since 2026-09-04 with a NEW BAR.** 2026-09-07 and
2026-09-08 both stood down on the identical 2026-09-04 tape (registry, calendar
finding). The 09-08 session moved things that matter: `^TNX` printed a fresh
252-day high (4.806), `DBC` a fresh 252-day high, `USO` +2.87%, `GLD` -1.73%,
`FXI` -2.45%, `^MOVE` +4.16%. So the price-state lane is genuinely fresh today,
which it was not on either of the two preceding mornings.

> **CORRECTION, filed 2026-09-09 after checker A (`a2c_c4_holiday.py`).** The
> first version of this map read "`^VIX3M` +4.43% against `^VIX` +2.75%" and it
> is WRONG. **`^VIX` carries a bar dated 2026-09-07, which was Labor Day**, and
> `^VIX3M`, `^SKEW`, `^MOVE`, `SPY` and `^GSPC` do not — only 50 of 1107 tickers
> have that bar and the rest are FX, futures, crypto and foreign indices. On the
> common equity calendar the 09-08 session was **`^VIX` +8.19%** (14.53 -> 15.72)
> against `^VIX3M` +4.43%, so the **VIX/VIX3M ratio ROSE from 0.8251 to 0.8548
> and the curve FLATTENED**. Spot vol was bid about twice as hard as three-month,
> which is the opposite of what candidate C4 was built on, and C4 died on the
> premise. `^VIX` has four such bars absent from SPY's calendar (2026-05-25,
> 06-19, 07-03, 09-07), all NYSE closures, so **any one-day `^VIX` change taken
> on `^VIX`'s own calendar is wrong on the session after each.** `00_recon.py`
> reindexes `^VIX` to SPY and is unaffected: the rel-range percentile of 3.57
> stands, and so does every watchlist verdict that uses it.

## Live state, the numbers every verdict below is against

| reading | value | basis |
|---|---|---|
| SPY | 765.96, -1.53% off 252d high | tape 09-08 |
| SPY 21d realized vol | 8.11% ann, **6.3rd pctile** of trailing 252 | recon |
| VIX / VIX3M | 15.72 / 18.39, ratio 0.855, UP from 0.825 on 09-04 (curve flattened) | tape, corrected |
| VIX 21d rel-range pctile | **3.57** (production VRC dial reads 1st) | recon |
| ^TNX | 4.806, **AT a 252d high** | recon |
| DBC | 32.40, **AT a 252d high** | recon |
| TLT / IEF / LQD vs 252d low | +1.43% / +0.34% / +0.25% | recon |
| HYG vs 252d high | -0.46% | recon |
| USO 21d | +23.78%, PIT rank 88.9 | recon |
| XOP / VLO | both AT 252d highs | recon |
| GLD / SLV vs 252d high | -19.40% / -43.78% | tape |
| sector 21d max-min spread | 18.45pp, 90.6th all-history pctile (81.3 trailing-252) | recon |
| DX-Y.NYB 63d rank | 14.7 | tape |
| fragility ma10(63d) | **87.9** as of 09-08 | state |
| P/C fear | off, 38.5th pctile | state |
| cycle | midterm (2026 % 4 == 2) | state |

## 1. Calendar x asset class

Seven events in the [-5,+15] td window: nfp 09-04 (-2), **ppi 09-10 (+1)**,
**cpi 09-11 (+2)**, **fomc_decision 09-16 (+5)**, **vix_expiry 09-16 (+5)**,
**opex 09-18 (+7)**, **quad_witching 09-18 (+7)**. Note vix_expiry and the FOMC
decision fall on the SAME date, which is unusual and is cell 1.8 below.

| event | class | verdict |
|---|---|---|
| ppi 09-10 | us_large | **DISMISS — registry-dead.** Both orderings of a one-session-apart PPI/CPI pair are measured; PPI-then-CPI is the negative side (SPY -0.1135% on N=127, gate worth -0.115pp where its complement is +0.024pp, placebo rank 8 of 11). Watchlist 42, arm needs permutation P<0.05 and the corrected September P is **0.7354** (`docs/answer_quality_review_2026-09-08.md`). |
| ppi 09-10 | rates | **CHECK (C12).** The pair gate is dead on duration too (Sept TLT -1.120pp on 12). But the pair gate is not the only conditioner available: the print lands with the ten-year AT a 252-day high, which is a rates-state condition nothing has crossed with a print anchor. Distinct from watchlist 41, which anchors on the COMMODITY high and died on its placebo ladder. |
| ppi 09-10 | commodities/energy | **DISMISS — registry-dead 09-08.** "Long commodities at a 252-day high into an inflation print" does not exist outside 2007/2008/2021/2022 (26 of 53 episodes, >100% of total; ex-those DBC -0.161%, USO -0.225%), placebo rank 7 of 11 (DBC) and 9 of 11 (USO), and the exact live both-prints-inside configuration is 13 episodes at -0.027%. Energy EQUITY leadership with a print in the hold: XLE -0.134pp, XOP -0.292pp, XOP positive at 0 of 10 horizons. |
| ppi 09-10 | dollar_fx | **CHECK (C5).** Dollar cells here are parked on cost (watchlist 15) or on the cycle year (watchlist 23), but neither is CPI/PPI-conditioned. The live configuration — DX-Y 63d rank 14.7 while the ten-year is at a 252d high — is the divergence itself. |
| ppi/cpi | gold_miners, metals | **CHECK inside C2/C3.** GLD pre-CPI as a bare cell is registry-dead (+0.040% vs its own +0.092% h=2 drift, 2026-08-07). The live angle is not "gold into CPI", it is gold REFUSING to confirm a crude-led inflation impulse. |
| ppi/cpi | credit | **CHECK (C11).** HYG 0.46% off its 252d high on a day the ten-year makes a 252d high. Watchlist 24's arm (SPY >= 2% off, dial < 50) fails on both legs today, but that entry conditions on SPY's depth; this cell conditions on RATES. |
| ppi/cpi | vol | **CHECK (C4).** Watchlist 33's arm is PASS (below), but its own settled work says the runway conditioner is what matters and PPI at runway 1 is disqualified. What is not settled is the term-structure move: VIX3M +4.43% against VIX +2.75% the session before the cluster. |
| ppi/cpi | us_small, intl | DISMISS. IWM inherits the pair kill (-0.193pp, 50.4% hit). Intl has no print-anchored history worth a slot beside C6, which is stronger on its own price state. |
| fomc 09-16 | all 15 classes | **DISMISS — registry-dead 09-01.** The pre-FOMC window was swept across fifteen asset classes; the family's fixed-effect common excess is -0.274pp at z -2.51. Separately, midterm years INVERT the Lucca-Moench drift (event-sleeve prereg), and 2026 is midterm. The short form is the Event Sleeve's own T2. |
| vix_expiry 09-16 | vol | **CHECK (C8), and only because of the collision.** VIX-expiry-week drift is registry-dead (+0.065% within-month paired excess, t 0.67) and the expiry/opex anchors are one anchor. What is NOT measured is a VIX settlement landing ON an FOMC decision date. |
| opex + quad 09-18 | us_large, us_small | **DISMISS — registry-dead 09-04.** "September quad witching is an FOMC anchor in costume": the run-in splits +2.382% (11 years, FOMC inside) against -0.834% (15 without), the laggard gate is worth +0.006pp, reference class puts IWM 6 of 16. This year HAS an FOMC inside, so the live subcell is the good half — but that subcell IS the pre-FOMC drift, which is midterm-inverted and is the sleeve's T2. Two independent blocks. |
| nfp 09-04 (-2) | rates | DISMISS. Watchlist 0 (midterm-dead, arm 2027-01) and watchlist 37 (needs prior surprise in (-100k,-50k] and a CPI inside h=3; prior print was -103k). Both fail; and the anchor is 2 td behind us. |
| election 11-03 | all | DISMISS. 39 td out, far beyond the 10 td horizon cap. |

## 2. Tape extremes, by class

- **energy — the loudest thing on the tape.** USO +23.78% 21d (rank 88.9),
  DBC/XOP/VLO at 252d highs, XLE 21d rank 92.5, OIH 63d rank 20.6 against XOP's
  77.0. Verdicts: the print-conditioned long is registry-dead (above); the
  OIH-vs-XOP services/E&P spread was killed 2026-08-25 (wrong-signed at h=1,2,3,5,
  -1.7x a two-leg round trip); the crude-thrust fade is watchlist 7 (needs >= 8
  post-2020 episodes, has 4) and the XLE thrust-band long is watchlist 4 (arm fired
  in full 09-02 and the cell died on four grounds). **CHECK as C2 and C3**, both of
  which use energy as a CONDITIONER on another class rather than as the leg.
- **rates — ^TNX at a 252d high, TLT/IEF/LQD pinned near 252d lows.** Watchlist 5
  needs TLT within 0.5% of its low (it is +1.43%, FAIL). Watchlist 30 needs ^MOVE
  level pctile in [40,50) (it is 72.2, FAIL). Watchlist 16 needs a >= +1.5% TLT
  day (it was -0.01%, FAIL). **CHECK as C1 and C12.**
- **vol — the pin.** SPY 21d realized vol at the 6.3rd pctile of its own year with
  the sector cross-section at the 90.6th all-history pctile. Cross-sectional
  dispersion as a DIRECTIONAL index signal is registry-dead and wrong-signed
  (short pays -0.649% at h=10 over 369 episodes; high dispersion is followed by
  SPY UP). So the dispersion leg may not be pitched as direction. **CHECK as C1
  and C4** on the VOL side rather than the direction side.
- **credit — HYG 0.46% off its 252d high**, LQD 0.25% off its 252d LOW. The IG/HY
  split is watchlist 1 (4 episodes, needs 8) and watchlist 26 (ONE episode in all
  of history). **CHECK as C11**, which is the rates-conditioned form neither entry
  covers.
- **gold and miners — GLD -19.40% off its 252d high, GDX -15.05%, GDX 21d +9.48%
  against GLD +0.31%.** Watchlist 3 needs GDX 5d rank >= 95 (it is 42.1) plus
  GLD 63d rank >= 50 and GLD within 10% of its high (-19.4%): FAIL on three legs.
  **CHECK as C2 and C3** on the metals-vs-energy axis instead.
- **other metals — SLV -43.78% off its 252d high while +59.55% over 252d.** The
  complex-break continuation short is watchlist 29 (blocked on a lag profile one
  session wide). **CHECK inside C3.**
- **dollar and FX — DX-Y 63d rank 14.7, 5d rank 10.7, and NOT confirming the
  yield high.** Watchlist 15 (cost, needs 7.5 bps on the magnitude form) and
  watchlist 23 (parks to a non-midterm year, 2027) both fail. **CHECK as C5**,
  which is print-conditioned and so is neither.
- **international — EWZ 5d rank 96.0 at z10 +1.96, EEM 5d rank 77.0, FXI -2.45%
  on the day at a 21d rank of 28.2, EFA within 0.96% of its high.** Commodity EM
  ripping while China breaks is not in the registry. **CHECK as C6.**
- **us_large / us_small — nothing extreme.** SPY 21d rank 20.2, IWM 13.9, DIA
  9.1, QQQ 30.6. The index is quietly soft while the cross-section rages, which is
  the whole subject of C1 and C9.
- **cross-section — ten names at a 21-day rank at or below 2.4** (ROST, TJX, SYK,
  AON 0.4; SNA 0.8; HON 1.2; GD 1.6; HRL 1.6; ITA 2.0; CSCO 2.4). A SUBGROUP
  basket of these is registry-dead as of yesterday (the short-term reversal factor
  wearing a group label; broad-universe parent +0.537% over 1,024 episodes at
  t +4.26). **CHECK as C9** in its only untested form, the COUNT as a breadth
  statistic conditioning the INDEX.

## 3. Seasonal and cycle cells

`seasonality.board_candidates` carries no A/B tradeable setup; its five entries
are all regime CONTEXT (midterm-year book de-risking, and a P/C reading dated
2026-08-04 that the live P/C at the 38.5th pctile has since superseded).
September in a midterm year conditions everything above rather than standing as
an idea: it blocks the pre-FOMC long, the quad-witching run-in, watchlist 0,
23 and 27, and it is the reason C1 and C12 must both be split on cycle year.
No standalone seasonal cell is opened today.

## 4. Watchlist — verdict on every active entry (43)

Every entry is PASS unless marked. Values cited are today's.

| # | entry | verdict |
|---|---|---|
| 0 | TLT from the NFP close, long end at its floor | PASS — arm is the first non-midterm NFP, 2027-01. |
| 1 | LQD vs HYG at joint 52w extremes | PASS — needs >= 8 declustered episodes; 4. |
| 2 | SVXY overnight into CPI | PASS — arm is a beta-neutral drop-best-year edge of 40-50 bps; 19.7. No new CPI since parking. |
| 3 | GLD on a miner-led thrust | PASS — GDX 5d rank 42.1 (needs >= 95), GLD -19.40% off its high (needs within 10%), GLD 63d rank 36.9 (needs >= 50). Fails three of four. |
| 4 | XLE on a crude one-day thrust in [5,6)% | PASS — USO 1d +2.87%, outside the band. |
| 5 | TLT with the IG complex at 52w lows | PASS — TLT +1.43% off its 252d low, needs <= 0.5%. IEF (+0.34%) and LQD (+0.25%) clear; TLT is the failing leg. |
| 6 | SPY on a skew spike alone | PASS — ^SKEW 5d rank 50.0 (needs >= 95); and midterm blocks it anyway, a block that STEEPENED under yesterday's re-test (>= 98 pays -0.387pp). |
| 7 | Fade a crude thrust out of a deep base | PASS — USO 63d rank 64.3, needs <= 20. Not live regardless of the episode-count arm. |
| 8 | IHI medical-device thrust | PASS — IHI 5d -5.9%, the opposite state. |
| 9 | FXI five-day break inside an intact thrust | PASS — FXI 5d rank 34.9, needs <= 20; 63d rank 75.8 does clear. One leg short. |
| 10 | TLT on the November month-position effect | PASS — parks to a date, 2026-11. |
| 11 | Short SPY at a 52w high with the long end at a 52w low | PASS — SPY -1.53% off (needs within 0.5%). Arm is cost on the de-concentrated form regardless. |
| 12 | SPY on a vol pop inside a calm tape | PASS — VIX 21d rank 50.0 (needs <= 25), VIX +2.75% (needs >= 5%). |
| 13 | Gold on an unconfirmed rate rise | PASS — arm is a yield MAGNITUDE floor, not the rank; and the metal is 19.4% off its high. Adjacent to C2, which is the ENERGY-led version and takes the opposite side of gold. |
| 14 | XLK vs XLV after a rotation gap | PASS — needs three new subclass episodes outside the 2026 cluster; none since parking. |
| 15 | Short the dollar on an unconfirmed rate rise | PASS — arm is 7.5 bps on the magnitude-floor form. Live state qualitatively matches (TNX 21d rank 78.6, DX 21d rank 27.4), which is why C5 takes the print-conditioned form instead. |
| 16 | Short TLT after a big up day from the low zone | PASS — TLT 1d -0.01%, needs >= +1.5%. |
| 17 | KRE vs XLF on a breadth washout | PASS — arm is +0.35% at h=3 ex-crisis; no new episodes. |
| 18 | IEF vs TLT curve position | PASS — arm is the year's yield thrust; re-armed 09-01, nothing since. |
| 19 | Narrow energy thrust cluster, 2-3 names at z10 >= 2 | PASS — inside the entry's own 11-name complex the live count at z10 >= 2.0 is below the [2,3] band (VLO is the only member near it). |
| 20 | New-high breadth with the index off its high | PASS — needs SPY > 2.0% off (it is 1.53%) AND raw-21d fragility <= 50 (it is 65.9). Both fail. |
| 21 | Sector washout into a 52w high as a family | PASS — arm is heterogeneity, a method arm, unchanged. |
| 22 | XLU washout with the long end hit alongside | PASS — TLT 21d rank 52.8, needs < 25. |
| 23 | Bare dollar washout | PASS — parks to a non-midterm year. DX 21d rank 27.4, above the <= 2 rung anyway. |
| 24 | HYG at a fresh 52w high while the index has not | PASS — needs SPY >= 2.0% off (1.53%) AND dial < 50 (87.9). Both fail, the dial badly. C11 is the rates-conditioned cell, not this one. |
| 25 | Semis' deep correction | PASS — SMH 63d rank 4.0 is the live reading on the wrong side, the same number the entry names. |
| 26 | IG at 52w lows while HY prints a high | PASS — ONE episode in all history (2026-08-03); state is broadly live again but there is still nothing to measure. |
| 27 | IEF out of the Jackson Hole close | PASS — midterm-blocked to 2027-08. |
| 28 | The laggard that is still falling | PASS — arm is a live reading on the wrong side; unchanged. |
| 29 | Short silver after a complex break | PASS — arm is a mechanism for a one-session-wide lag profile; none offered. Related state IS live (SLV -43.8% off) and C3 takes the other side of it. |
| 30 | Long duration, yield high, bond vol MID-range | PASS — ^MOVE level pctile 72.2, needs [40,50). The yield-high leg is live; the MOVE leg is not. |
| 31 | Small-cap month-end overnight in December | PASS — parks to December. |
| 32 | Energy at a fresh 52w high on a down-SPY session, h=21 | **LIVE-ADJACENT and PASS.** XOP made a fresh 252d high on 09-08, a session SPY fell 0.55%. But the arm needs an h=21 family permutation P < 0.05 (it never goes under 0.0510), h=21 is outside this product's 10 td cap, and the standing blocker is an INVERTED dose response that may not be waived. |
| 33 | SVXY into a print out of a compressed VIX range | **PASS on the number that matters.** The arm is a rel-range pctile in **(5.0, 15.0]**; today reads **3.57**, inside the (0,5] bucket the entry itself measured as DEAD (-0.096% over 25 anchors, 13-11, against +1.465% and +2.034% in the live bands). CPI 09-11 has the runway (3) but not the band. |
| 34 | Pooled sector triple rank floor | PASS — arm is a reason to exist beside the book, which nothing has supplied. |
| 35 | SPY into a print out of a dead VIX range | **PASS on the dial.** rel-range 3.57 clears <= 15, but the arm also needs ma10(63d) BELOW 50 and it is **87.9**, the 95th percentile of its own series, with zero of the 15 gated anchors carrying a dial reading above 70. |
| 36 | Risk premium across an extended closure | PASS — the closure is behind us and the arm is a pre-registered forward test. |
| 37 | Post-NFP duration after a moderate prior miss | PASS — prior print was -103k, outside (-100k,-50k]. |
| 38 | SVXY at the first close after a closure | PASS — the entry itself notes 2026-09-08 does not qualify (runway 2), and today is k=+2. |
| 39 | SPY vs IWM in the dial's [56,70) band | PASS — dial 87.9, needs a fall of at least 18 points. |
| 40 | HYG at the first close back from a closure | PASS — needs HYG > 1% below its 252d high (it is 0.46%) or 21d rvol above 4.4%. The anchor session has also passed. |
| 41 | Short IEF with commodities at a 252d high and a print in the hold | **STATE IS LIVE, PASS on the arm.** DBC is at a 252d high and both PPI and CPI sit inside an h=5 hold. The arm is the placebo ladder — the true print anchor ranks 5 of 11, k=-5 and k=-3 both paying more — and nothing about that has moved. C12 anchors on the RATES state instead, which is a different conditioner. |
| 42 | SPY across the September PPI-then-CPI pair | **PASS, and retired in substance.** The arm needs the 12-month permutation P below 0.05; the corrected value for September's own threshold is **0.7354**, not the 0.1618 the entry quotes, and the parent gate is still negative. |

None expired; none fired. Two (33, 35) are the closest and both fail on a single
named number, which is what those entries exist to do.

## 5. Scoreboard read before selecting

Five graded ideas lifetime, avg +0.174R, hit 80%. By axis: `event_fingerprint`
2 at +0.622R, `interaction_cell` 1 at +0.146R, `relative_value` 1 at +0.099R,
`inversion` 1 at -0.620R. By grade: B 3 at +0.448R, C 2 at -0.237R. The graded
count is a handful, so no axis is penalised or favoured on it today. The one
signal worth respecting is grade discipline: both C ideas are underwater and
both B ideas are not, which argues for at most one C and for it to earn the slot.

## 6. Candidates selected (12)

| # | candidate | axis | classes |
|---|---|---|---|
| C1 | Yields at a 252-day high while index realized vol sits in its bottom decile | interaction_cell | rates, us_large, vol |
| C2 | A crude-led inflation impulse that gold refuses to confirm | interaction_cell | energy, metals, rates |
| C3 | Commodity index at a 252-day high with its metals leg in a deep drawdown | relative_value | metals, energy |
| C4 | Three-month vol bid while spot vol stays dead, the session before a print cluster | event_fingerprint | vol, us_large |
| C5 | The dollar at a 63-day rank floor into an inflation print | inversion | dollar_fx, event |
| C6 | Commodity EM ripping while China breaks | relative_value | intl |
| C7 | Nearest-neighbour tapes to this configuration | historical_analogue | multi |
| C8 | A VIX settlement landing ON an FOMC decision date | flow_mechanics | vol, event |
| C9 | Breadth of 21-day rank floors as an index conditioner | interaction_cell | us_large, cross-section |
| C10 | Vehicle choice for whichever of C1/C4 survives | instrument_translation | us_large, us_small, vol |
| C11 | Credit at a 252-day high while the ten-year prints one too | interaction_cell | credit, rates |
| C12 | A PPI print landing with the ten-year already at a 252-day high | event_fingerprint | rates, event |

Six axes. Ten asset classes touched (us_large, us_small, rates, credit, gold and
miners, other metals, energy, dollar and FX, international, volatility).
Event-anchored: C4, C5, C8, C12. Price-state-anchored: C1, C2, C3, C6, C9, C11.
Both search modes are crossed in C1, C5 and C12, which is the 2026-08-07 lesson.
