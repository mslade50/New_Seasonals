# Surface map, 2026-09-15 (Tuesday)

Inputs: `data/pitch_state.json` generated 05:11 ET; `data/pitch_tape.json` 218 names,
freshest bar 2026-09-14 (Monday, the prior session; only LEG stale at 08-27,
irrelevant). Pipeline green (4/4 receipts). Fragility dial ma10(63d) 85.2 as of
2026-09-14 (PIT append-only vintage; raw 63d 78.5, raw 21d 49.5, raw 5d 19.3).
That is market context for each idea's own risk, not a sizing instruction, and
not a historical maximum. Equity P/C fear OFF (59.5th pctile, data 09-14).
Signals on: VIX Range Compression only. Cycle: midterm year, September. Live
probe numbers come from `00_live_arms.py` (output `00_live_arms_out.txt`) unless
quoted from the tape file. Entry for anything shipped today is the 2026-09-15
MOO or MOC (lag=1 off the 09-14 close); h=1 ends on the 09-16 FOMC decision
close, which is also the VIX settle date; h=3 ends at the 09-18 opex / quad
witching / S&P rebalance close; h=10 ends 09-29.

Scoreboard read: 5 graded ideas lifetime (B 3 at +0.448R, C 2 at -0.237R;
event_fingerprint 2 at +0.62R, inversion 1 at -0.62R). A handful, not a signal.
Twelve straight stand-downs (08-28 .. 09-14). Two brief notes carried into the
checker prompts: (1) a gate that does not filter removes attribution, it does
not by itself kill a live parent; (2) a HOMOGENEOUS positive family effect does
not kill a member trade, it re-prices the member at the shrunk family estimate,
which must then clear cost on its own.

Portfolio/strategy overlap is NOT assessed (owner decision 2026-09-08).

## What changed on the 09-14 tape

A violent intra-market rotation under a calm index: SPY -0.45%, but SMH -4.75%
(-1.5 ATR, AMAT -7.07%, MU -5.25%, AVGO -4.77%), money-centre and custody banks
down hard while regionals rose (BAC -5.14% = -2.93 Wilder ATR, GS -3.96%, MS
-3.64%, BNY -3.07%, STT -2.71%; KRE +0.28%, KEY +0.41%), oil services broke
while crude rose (OIH -4.42%, SLB -4.89%; CL=F +10.83% over 5d, OIH -5.96%
over 5d), metals and miners fell (XME -7.12% 5d, GDX -3.05% on the day),
last week's single-name thrusters reversed (GLW -13.70% = -2.69 ATR after
+13.97% in 5d; INTC, AMD, ADI about -1 to -1.6 ATR), and defensives bid (XLV
+1.45%, XLP +1.25%, XLV minus XLK one-day gap +3.25pp). ^VIX +7.95% to 17.10
after -11.2% on 09-11 (2d -4.15%); VIX/VIX3M 0.887. ^TNX 4.961, 0.28% under
its 252-day max, +17.7 bp over 5d and +32.0 bp over 21d; ^MOVE 83.9 at the
93.3rd trailing-252 level percentile and r5 92.9.

## 1. Calendar events x asset classes

Events in window: PPI 09-10 (-3), CPI 09-11 (-2), FOMC decision 09-16 (+1),
VIX expiry 09-16 (+1, a collision), opex + quad witching 09-18 (+3), NFP 10-02
(+13). Not in the event file but live: S&P quarterly rebalance at the 09-18
close, quarter-end 09-30 (+11). Earnings inside 10 td: FDX 09-17, GIS and PAYX
09-23, COST 09-24.

| event | us_large | us_small | rates | credit | gold/miners | other metals | energy | dollar/fx | international | volatility |
|---|---|---|---|---|---|---|---|---|---|---|
| PPI 09-10 (-3) | dismiss: Sep PPI/CPI pair killed 09-08 (W42; corrected perm 0.7354) | dismiss, same | dismiss: W43 release-close anchor passed | dismiss: no credit PPI cell | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| CPI 09-11 (-2) | dismiss: CPI-session SPY at TNX high killed 09-11 | dismiss | dismiss: CPI x TNX-high killed 09-11 | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss: W33 second rung anchored on the 09-11 close, passed |
| FOMC 09-16 (+1) | dismiss: pre-FOMC window swept across 15 classes 09-01 (Cochran p 0.81); midterm run-in wrong-signed (W47); the index leg of C3 is folded into C3 as the vehicle question | dismiss: IWM run-in collision gate negative (reg 09-11) | CHECK as C6: bond vol AND yields both thrusting into the decision (MOVE r5 92.9, TNX r5 97.6). Distinct from the 09-01 kills, which conditioned on the yield level alone and on bond vol popping while equity vol FELL | dismiss: in the 15-class sweep, and HYG/IEF r5 91.3 says this week's HYG dip is duration | CHECK as C8: short gold into the decision after a yield-thrust week, the INVERSION of the 09-14 kill (GLD 2-7 at -1.633%); charged as a post-hoc sign flip | dismiss: SLV in the 15-class sweep; W29 lag-profile arm unmet | dismiss: crude into midterm FOMC dead (reg 09-01, placebo 8 of 12) | dismiss: DX in 15-class sweep; DX r21 44.4, no rate-vs-dollar arm live | dismiss: EFA/EEM/FXI in 15-class sweep | CHECK as C3: the pre-decision premium RE-BID (VIX +7.95% at k=-2 after a crush at k=-3) sold into the decision. 09-14 showed a crush is not re-bid; this is the other state |
| VIX expiry 09-16 (+1) | dismiss: collision is the FOMC anchor (reg 09-01, 09-11) | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | folded into C3: W45 settle-session SVXY on collisions is -0.297% on 2018+; C3 must report the collision subset separately and may not lean on it |
| opex/quad 09-18 (+3) | dismiss: Sep quad run-in is an FOMC anchor in costume (reg 09-04, 09-11) | dismiss: short IWM into Sep quad killed 09-11; post-quad window starts 09-18 close, not enterable | dismiss: no duration mechanism at an equity expiry | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss: V4 post-opex crush excludes September by prereg; entry 09-18 not enterable |
| S&P rebalance 09-18 | dismiss: index-add flow is a real mechanism but add/delete history is not in the repo, so it cannot be falsified locally (honesty rule) | dismiss, same | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| NFP 10-02 (+13) | dismiss: outside a 10 td hold | dismiss | dismiss: W37 fires only on the print | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| quarter-end 09-30 (+11) | dismiss: QE rebalancing TLT/SPY killed 09-14 (no QE premium, +0.218% vs +0.233% ordinary ME) | dismiss | dismiss, same kill | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss | dismiss |
| FDX earnings 09-17 (+2) | dismiss: pre-print drift in a deeply lagging name killed 08-31 (gate monotone against lagging names; FDX r21 2.4, r63 0.8 is the worst rung); earnings anchors 12-for-12 on the placebo ladder | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| midterm September (cycle) | dismiss: second-half midterm September short killed 09-14 (2002/2022 carry 120%, above-200d 1-3) | dismiss, same | dismiss: TLT month-position parks to November (W10) | dismiss | dismiss | dismiss | dismiss | dismiss: W23 midterm-inverted | dismiss | dismiss |

## 2. Tape extremes by class (close 2026-09-14)

- **rates**: ^TNX 4.961 (-0.28% under the 252 max 4.975), r5 97.6, r21 96.8, z10 +1.90, +17.7 bp 5d. TLT 0.19% above its 52w low, IEF and LQD exactly at theirs, IEF r5 1.2. ^MOVE +14.8% 5d, level pctile 93.3. W5 and W18 both already killed on this state (09-11, 09-14). CHECK C6 on the vol-plus-yield thrust into the decision only.
- **credit**: HYG r5 4.4, z10 -1.80, but HYG/IEF r5 91.3: duration again, and the 09-14 trap says rate-driven HYG flushes pay below drift (+0.070% vs +0.104%). LQD at its low is duration (W1: LQD residual on IEF +0.000pp). Dismissed.
- **us_large / sectors**: SPY -2.19% off high, r5 18.3, 6.7% above its 200d; breadth 23.4% of the tape above a 21d rank of 50. XLI 5.6/2.0/1.6 and XLU 6.3/4.0/4.0 at r5/r21/r63; ITA r21 1.2. Dismissed: the 22-ETF washout family pays -0.26pp with SPY above its 200d (reg 09-14), ITA and XLI washouts killed. Utilities: W22 misses by 0.79 rank points (below). Banks: BAC -2.93 ATR on a day KRE rose. CHECK C1. Healthcare bid (XLV +1.45%) after last week's flush: dismissed, W49 parked on a cost arm and the XLV-minus-XLK one-day rotation family is dead in four expressions (reg 08-19, 08-25).
- **tech/semis**: SMH -4.75% (-1.5 ATR), r5 12.7, r21 6.0, r63 1.2, 252d return +78.7%. W25's still-falling arm (r5 < 15) is satisfied for the first time; its heterogeneity arm is not. CHECK C2 as the pooled family form (the conditioner tested out of sample on the 22 non-SMH members) and C7 as the unconditioned one-day industry-shock parent. Single-name thrust reversals (GLW -2.69 ATR, ADI -1.60, INTC/AMD about -1.0): CHECK C5. Software bid (ADSK +7.78%, ADBE +5.30%, CRM +4.73%): dismissed, no software ETF in master_prices and the one-day rotation family is dead.
- **us_small**: IWM r21 6.3, r63 3.6, -5.63% off its high. Dismissed: both IWM/SPY pair directions closed (09-11, 09-14); IWM into quad killed.
- **gold/miners**: GLD -3.42% 5d, -20.8% off high; GDX -5.16% 5d, -3.05% on the day. W3 (miner-led thrust) wrong-signed. Miners-vs-metal family dead (09-02, 09-07). CHECK C8 (event form only).
- **other metals**: SLV -4.98% 5d, -46.2% off high, 1d -2.20% against W29's -4.00% depth arm. XME -7.12% 5d, r5 9.1, r63 9.5: dismissed, XME sits in the 13-name subsector washout class that pays nothing above SPY's 200d; metals-vs-energy at a commodity high killed 09-09.
- **energy**: CL=F 101.39, +10.83% 5d, r5 92.1, but -10.23% below its own 252 high (so USO's -1.09% from high is roll yield, reg 09-14 trap). OIH r5 5.2, -5.96% 5d, -4.42% on the day; SLB -4.89%. W19 count 0 under pitch_lab.zscore (USO 1.64). CHECK C4 on the services-versus-crude split, defined on the front contract. Refiners (VLO +48.6% 63d) dismissed: crack-spread data absent.
- **dollar/fx**: DX-Y.NYB r21 44.4, 5d +0.29%, flat through +32 bp of 21-session yield rise; UUP r5 58.3. Dismissed: W13 needs DX r21 <= 15, W15 <= 20; re-deriving a looser dollar rung is the anti-rescue pattern.
- **international**: EEM -2.73% on the day, r5 7.9, r63 3.2 (the semis weight in EEM, same factor as C2/C7); FXI +1.01%; EWJ -0.99% off its high; EWZ r21 85.3. Dismissed: the country-decoupling family is dead in both directions (EWZ twice, FXI, EFA, the EM funding form), and today's EEM move is the SMH move with an EM label.
- **volatility**: ^VIX 17.10, +7.95% on the day after -11.2% on 09-11; VIX 21d relative range at the 20.6th trailing-252 pctile (the dashboard's own definition reads 13th); VIX/VIX3M 0.887; ^SKEW r21 98.4. CHECK C3. SKEW forms dead (W6 midterm-blocked; 21d SKEW rank killed 09-08 in three expressions).

## 3. Seasonal and cycle cells

- Seasonal board payload is stale (asof 2026-08-05, 0 A+B setups); not used as evidence.
- Midterm September: short killed 09-14. Midterm is the conditioner that blocks W0, W6, W23, W27, W47, and C3/C6/C8 must each report their midterm split.
- NG=F September (W46): mechanism arm unmet (September ranks 5 of 12). Dismissed.

## 4. Watchlist verdicts (51 active, 0 expired)

- W0 TLT NFP non-midterm: PASS. Midterm; parks to 2027-01.
- W1 LQD/HYG joint extremes: PASS. HYG -1.20% off high against within 0.5%.
- W2 SVXY overnight into CPI: PASS. Next CPI 10-14.
- W3 GLD on a miner-led thrust: PASS. GDX r5 19.4 against >= 95.
- W4 XLE on a crude one-day thrust 5-6%: PASS. USO 1d +1.14%.
- W5 TLT with IG complex at 52w lows: PASS. The state is live again (TLT +0.19% off low, IEF/LQD at lows) but the arm is a construction test (deleted-vs-kept gap >= 0.35pp, today +0.043pp), not a state.
- W6 SPY on a skew spike: PASS. Midterm; ^SKEW r5 50.0.
- W7 fade crude thrust from a deep base: PASS. USO r63 73.4 against <= 20.
- W8 IHI thrust: PASS, wrong-signed. IHI r21 11.5.
- W9 FXI break inside thrust: PASS. FXI r5 18.3 but r21 59.5 against >= 80.
- W10 TLT November: PASS. Parks to November.
- W11 short SPY at 52w high with TLT at low: PASS. SPY -2.19% off high against within 0.5%.
- W12 SPY vol pop in calm tape: PASS. VIX up 7.95% and SPY down 0.45% clear two legs, but VIX 21d rank 81.0 against <= 25.
- W13 gold on unconfirmed rate rise: PASS. DX r21 44.4 against <= 15.
- W14 tech vs healthcare rotation gap: PASS, and note the state FIRED on 09-14 (XLV-XLK +3.25pp, SPY -2.19% off high, SPY Wilder ATR 0.82%). The arm needs three new subclass episodes OUTSIDE the 2026 cluster; a 2026 episode cannot satisfy it. Standing note: if it ever arms it arms as an outright XLK long.
- W15 short dollar on unconfirmed rate rise: PASS. DX r21 44.4 against <= 20.
- W16 short TLT after a big up day near the low: PASS. TLT 1d +0.07%.
- W17 KRE vs XLF bank breadth: PASS. Ex-crisis cost arm; and today's bank move is the opposite shape (money-centre down, regionals up), which is C1.
- W18 IEF vs 0.523 TLT curve: PASS. Killed 09-14 on its own arm; out-of-sample only.
- W19 narrow energy thrust count: PASS. pitch_lab.zscore count 0 (USO 1.64) against [2,3].
- W20 new-high breadth, survivorship-free: PASS, and both numbered arms moved for the first time: SPY -2.19% off its high clears (a) (> 2.0% below) and raw-21d fragility 49.5 clears (b) (<= 50). The cell underneath the arms still does not fire: its breadth leg (c2_c6_breadth_attribution.py) is the PIT percentile >= 80 of the share of the nine SPDRs within 0.25% of a 52w high, and today ZERO of nine qualify (nearest XLE -1.19%, XLF -2.61%). Re-check on the first session a sector prints a fresh high while SPY stays more than 2% off its own; the dial caveat (population max 80.6 against 85.2) stands.
- W21 sector washout within 5% of high: PASS. No SPDR with r5 <= 5 (XLI 5.6) sits within 5% of its high (XLI -8.89%).
- W22 XLU washout with TLT hit: PASS, closest miss on the list. XLU r21 3.97 clears <= 5; TLT r21 25.79 against < 25, short by 0.79. Not re-derived at a looser rung (anti-rescue).
- W23 bare dollar washout: PASS. Midterm.
- W24 HYG fresh high while index not: PASS. HYG -1.20% off high.
- W25 SMH 63d floor in a top-decile year: PASS on the arm as written (the 23-ETF heterogeneity leg is unmet), but the still-falling leg is live for the first time (r63 1.19, r5 12.70 < 15). CHECK as C2 in its family form.
- W26 IG lows with HY at a high: PASS. HYG -1.20% against within 0.25%.
- W27 IEF post-Jackson Hole: PASS. Midterm.
- W28 laggard still falling, pooled: PASS. No name holds r21 >= 90 AND r63 <= 10 (SMH r21 6.0).
- W29 short SLV after complex break: PASS. SLV 1d -2.20% against the -4.00% depth bucket; the lag-profile mechanism is unaddressed in any case.
- W30 duration at yield high with MOVE mid-range: PASS. MOVE level pctile 93.3 against [40,50).
- W31 IWM December month-end overnight: PASS. Parks to December.
- W32 XLE fresh high on a down-SPY session, h=21: PASS. XLE -1.19% off high, no fresh high.
- W33 SVXY into a print out of a compression band: PASS. Dial 85.2 against <= 68.0; the k=-2 FOMC anchor was the 09-14 close.
- W34 pooled sector triple floor: PASS. Restated live-state arm (a nine-SPDR member at the 5/21/63 floor) is arguably live on XLI (5.6/2.0/1.6) and XLU (6.3/4.0/4.0), but the 09-14 finding binds: the washout family pays -0.26pp with SPY above its 200d, and SPY is 6.7% above.
- W35 SPY into a print out of a dead VIX range: PASS. Dial 85.2 against < 50.
- W36 closure risk premium: PASS. No closure.
- W37 post-NFP duration, moderate prior miss: PASS. Next NFP 10-02.
- W38 SVXY first close after closure: PASS. No closure.
- W39 SPY vs IWM at dial 56-70: PASS. Dial 85.2.
- W40 HYG after closure: PASS. No closure.
- W41 short IEF, commodities at a 252 high with a print in hold: PASS. DBC -1.40% off high; next CPI 10-14.
- W42 SPY across the Sep PPI-CPI pair: PASS. Printed.
- W43 TLT from the PPI release close: PASS. Anchor passed.
- W44 SPY with HYG at high on a TNX-high session: PASS. HYG -1.20% off high.
- W45 SVXY settle session on an FOMC/VIX-expiry collision: PASS. 09-16 IS a collision and the settle session is tomorrow, but the arm is 16 post-2018 collisions (today's would be the 13th); record tomorrow's outcome. C3 reports its collision subset and may not use it.
- W46 NG=F September: PASS. Mechanism arm unmet.
- W47 SPY run-in on a collision, non-midterm: PASS. Midterm.
- W48 deep 5d flush inside a top-decile 63d trend: PASS. No r5 <= 2 with r63 >= 90 among the ETFs (IBB r5 5.6 / r63 83.7).
- W49 XLV vs 0.71 SPY after a healthcare flush: PASS. XLV r5 12.3 against <= 1; the arm is a family cost threshold in any case.
- W50 beta-hedged short SVXY after a 10% VIX crush: PASS. 09-14 was a +7.95% VIX pop, not a crush.

## 5. Candidates selected (8), with axis and source cell

| id | class | anchor mode | axis | candidate |
|---|---|---|---|---|
| C1 | us_large (banks) | price-state | relative_value | Money-centre bank one-day shock with regionals up: a large bank (JPM BAC C WFC GS MS BNY STT) down >= 2 Wilder ATR on a session KRE closes >= 0 and SPY > -1%; long the name hedged against XLF, h=1..5. Live on BAC (-2.93 ATR, KRE +0.28%) |
| C2 | us_large (semis) | price-state (W25 live leg) | interaction_cell | The leader's 63d floor inside a top-decile year while STILL FALLING (r63 <= 5, 252d return top-decile, r5 < 15), pooled over the 23-ETF reference class and tested ex-SMH; long SMH h=10 at the family estimate |
| C3 | volatility | event (FOMC +1) | event_fingerprint | Pre-decision premium re-bid: ^VIX up >= 5% at k=-2 (live +7.95%), long SVXY MOC k=-1 to the decision close (h=1) or h=2, with the mandatory SPY residual, collision subset reported |
| C4 | energy | price-state | relative_value | Oil services flush while the front contract thrusts: OIH r5 <= 10 with CL=F r5 >= 90 (live 5.2 / 92.1); long OIH outright and hedged against XLE or XOP, h=3..10 |
| C5 | us_large (single names) | price-state | inversion | Failed thrust: a name at a 5d rank >= 80 on t-1 that falls >= 1.0 Wilder ATR on day t while SPY is down less than 1%; long the name hedged against SPY, h=1..5, pooled over the 218-name tape universe. Live on GLW, ADI, INTC, AMD |
| C6 | rates | event (FOMC +1) x price-state | interaction_cell | Bond vol and yields thrusting together into a decision: ^MOVE r5 >= 90 AND ^TNX r5 >= 90 at k=-1/k=-2; long TLT (or IEF) MOC k=-1, h=1..3 |
| C7 | us_large (industry ETFs) | price-state | inversion | Industry-shock drift: a 23-ETF member falls >= 1.5 Wilder ATR on a session SPY falls less than 0.75%; SHORT the member against SPY, h=1..5 (the 09-14 hint that single-big-day floors keep lagging, tested out of sample on the family). Live on SMH and OIH. Shares the SMH/OIH state with C2/C4 in the opposite direction, so the checker charges the sign |
| C8 | gold/miners | event (FOMC +1) x price-state | inversion | Short gold into the decision after a yield-thrust week (TNX r5 >= 95 within 1% of its 252 max), the sign flip of the 09-14 kill (GLD 2-7 at -1.633%); short GLD (or GC=F) MOC k=-1, h=1..5, charged as a post-hoc flip |

Added during stage C (the counter-story C1's checker brief named before any
data was read, taken through round 2 as its own candidate and charged a
two-sided sign test):

| id | class | anchor mode | axis | candidate |
|---|---|---|---|---|
| C1x | us_large (banks) | price-state | relative_value | Continuation: short a large bank against XLF at beta after a non-earnings, intraday-led drop >= 1.5 Wilder ATR on a session SPY falls less than 1%, h=1..5. Live on BAC (gap share 0.11, break at 13:00 ET), GS, MS, BNY |

Coverage: 5 asset classes (us_large, volatility, energy, rates, gold/miners);
event-anchored C3, C6, C8; price-state C1, C2, C4, C5, C7; axes relative_value,
interaction_cell, event_fingerprint, inversion (4). us_small, credit, other
metals, dollar/fx and international were opened and dismissed above with the
numbers they turned on.
