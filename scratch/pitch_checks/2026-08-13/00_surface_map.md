# Surface map — 2026-08-13 (Thursday, PPI day, midterm year)

State: `data/pitch_state.json` (no warnings), freshest bar **2026-08-12** = prior
session, pipeline 7/7 green, dial as-of 08-12 (age 1 td), P/C as-of 08-12.
Tape sorted whole in `00_tape_sort.py`; asserted numbers in `00_map_facts.py`.

Regime one-liner: SPY 0.10% off its 52w high with z10 +2.16, breadth 72.9% above
the 200d, VIX 14.55 (63d rank 16), **fragility dial ma10-63d 72.2 against 29.5
twenty-one sessions ago**, exposure leg killed (raw-21d 73.5 > 50), P/C fear OFF
at the 44th percentile, Low Absorption Ratio the only signal on (2nd percentile).

Live pitch exposure that constrains today: **DX long to the 08-14 close**, **GDX
long to the 08-17 close**, **TLT long exits on today's 08-13 close**. Three
staged Overflow scanner SELLs (tickers not carried in state).

Axis feedback read: the scoreboard has **1 graded idea** (relative_value, +0.099R
on a 1-td QQQ/SPY pair). One observation is not a per-axis split, so no axis is
being up- or down-weighted today. Recorded and moved on.

---

## 1. Calendar events x asset classes

Six events sit in the [-5, +15] td window. Two are behind us (NFP 08-07 at -4,
CPI 08-12 at -1) and are dismissed as anchors: an entry tonight is after both,
and the CPI/PPI pair has been mined into the ground across the 08-10, 08-11 and
08-12 runs (11 kills between them). PPI is **today**, which means a MOC entry
tonight is *post*-print, so PPI is not an entry anchor either — it is an exit
anchor for yesterday's TLT idea and the left edge of a calendar vacuum (cell V).

Live forward anchors: **vix_expiry 08-19 (+4 td)**, **opex 08-21 (+6 td)**,
**jackson_hole 08-28 (+11 td)**. NFP 09-04 (+16), PPI 09-10 (+20), CPI 09-11
(+20), FOMC 09-16 (+23), quad witching 09-18 (+25) are all outside any legal
horizon (max 10 td) and are dismissed on that alone.

| class \ event | vix_expiry +4 | opex +6 | jackson_hole +11 |
|---|---|---|---|
| **us large** SPY QQQ ^GSPC ^NDX | **DISMISSED, registry** — "VIX-expiry-week drift" was swept 2026-08-07: the raw +0.175% (N=319) is mid-month position, within-month paired excess +0.065% at t=0.67, 2018+ paired excess negative, and the mechanism is falsified inside its own window because the settle day is the worst day of the week. Buying the settle-day short instead is the post-hoc-sign-flip trap, also in the registry | **DISMISSED, registry** — "the run into August opex" was swept 2026-08-07: +0.342% over 26 non-overlapping years against SPY's +0.374% unconditional h=10 drift, i.e. worse than a random 10-day long, with the whole effect in 2000-2004 (2010+ is -0.514%). The midterm restriction anti-works (SPY midterm +0.361% vs +0.531%) and "midterm mid-August seasonality" is separately dead at N=6 carried by 2002 | dismissed: an entry today exits at td 10 = the session before JH, so the JH *reaction* is unreachable; the run-up is cell C4 and SPY is its weakest leg (the 08-11 JH work already showed the index legs are carried by 2010-11) |
| **us small / breadth** IWM | dismissed with the us-large cell above | dismissed with the us-large cell above (IWM's midterm mid-August cell is the one carried by 2002) | dismissed: **killed 2026-08-11** — short IWM on the JH anchor is wrong-signed in midterms (+0.286%, 3 of 6 down) and dies on 3 dropped years |
| **rates** TLT IEF ^TNX | not examined: a VIX settlement is an equity-vol flow event with no channel to the long end; no mechanism to test | not examined: same, and the IG-complex duration cell already has its own live watch entry | CHECK — cell C4. Powell at JH is the one event in the window that speaks directly to the front end, and the long end sits 0.23% off its 52w low |
| **credit** HYG LQD | not examined | not examined | dismissed: 2026-08-11 killed the HYG event cell as a pre-2018 fossil (+17.8 bps pre-2018, -2.8 bps after) and 08-12 killed LQD's incremental value over IEF (residual -3.15 bps) |
| **gold / miners** GLD GDX NEM | not examined | not examined | CHECK — cell C4b, gold into JH. Live long-GDX exposure to 08-17 caps what can ship, and the metal legs were killed 08-10/08-11, but the JH anchor is a different question from the thrust anchor |
| **other metals** SLV XME | not examined | not examined | dismissed: SLV was killed 08-11 for doubling the live GDX leg (corr +0.708, alpha -0.666 at t -2.68); nothing about a JH anchor repairs a vehicle problem |
| **energy** USO UNG DBC XLE | not examined | not examined | dismissed: the entire energy surface was killed twice this week (08-11 crude-thrust XLE residual +0.291% at a 49.3% hit; 08-12 the USO fade is roll cost, era-flipped). A JH anchor cannot rescue a class whose price state is the problem |
| **dollar / FX** UUP DX-Y.NYB | not examined | not examined | CHECK — cell C4c, the dollar into JH, and it is the only class where a *live* position argues for the check rather than against it (DX long exits 08-14, so a JH-window trade starts after it is flat) |
| **international** EFA EEM FXI EWZ | not examined | not examined | not examined: no mechanism connecting a Fed conference to ex-US equity that is not just the dollar leg (C4c) restated |
| **volatility** ^VIX ^VIX3M ^MOVE SVXY | **DISMISSED, registry** — "pre-expiry short-vol carry (long SVXY into VIX expiry)" was swept 2026-08-07 (gate-matched control eats it, 2018+ +0.19% at t=0.18 on the -0.5x instrument, one gated window -24.8%), and "post-NFP, post-FOMC and post-VIX-expiry vol cells" are listed swept and empty. The proximity of the expiry survives only as a tail-risk note inside cell C1, not as its anchor | dismissed with the same registry entry: opex ex-September is the ONLY post-event vol cell that survived the 2026-08-06 sweep, and the event sleeve already owns it as V4 | dismissed: the vol path into JH is dominated by the two nearer expiries, so a JH-anchored vol trade cannot be attributed |

## 2. Tape extremes, by class

**us large.** SPY 0.10% off its 52w high, z10 +2.16 (8th highest in the tape),
5d rank only 45.6 — a grind, not a thrust. DIA is the anomaly: 5d rank 16.7 and
-1.04% while SPY is +0.35%, yet DIA's 63d rank is 83.3 against SPY's 44.4.
→ **CHECK C10** (DIA/SPY divergence). QQQ 2.90% off its high with a 63d rank of
26.2 while SPY is at its high: the laggard-Nasdaq form of this was traded 08-11
and graded (+0.099R), and its fingerprint sits inside the 10-td repeat window,
so it is not re-pitchable without a `changed_since`. Dismissed as repetition.

**us small.** IWM printed a new 52w high (0.00% off) with a 21d rank of 49.6.
Nothing extreme. Folded into C3 only.

**rates.** The whole investment-grade complex is pinned: TLT 0.23% off its 52w
low, IEF 0.86%, LQD 0.75%, with ^TNX 5d rank 71.4 and 1.33% off its 52w yield
high. The tight rung of the **live watch entry (W6) is ON** but its freshness
leg is not — see section 4. Bond vol: **^MOVE's level sits at the 46.2nd
percentile of its trailing year**, so the "bond vol in its bottom decile at a
duration extreme" story is dismissed for the second time on the same number
(the 2026-08-10 kill measured 45.6). Not examined further.

**credit.** HYG at a 52w high (0.00%) while LQD is 0.75% off its low: the joint
extreme of watch entry W2, still the same cluster that began 07-22. PASS, see
section 4.

**gold / miners.** NEM has the tape's highest z10 at +3.04 and +12.99% in five
sessions; GDX +8.70% (5d rank 88.9, 21d rank 94.0); CEF +5.96%; GLD +3.92% (5d
rank 84.9). All of it sits on top of a 63d rank of 27 — a thrust out of a base.
Dismissed as an *anchor*: the 08-10 and 08-11 runs killed the miner-over-metal
spread (beta-neutral -0.000%), long SLV, long GLD alongside the live GDX leg,
and 08-12 killed the fade. The one live watch (W4) requires GDX 5d rank >= 95
while GLD's is < 95; today GDX is **88.9**, so it is not even live. Metals
appear today only as C4b's JH leg.

**other metals.** SLV is the loudest number on the tape (63d -24.81%, 63d rank
6.0, 44.07% below its 52w high) and it is catching up (+11.08% over 21d). The
obvious construction is a ratio trade, and **the ratio is not extreme**:
GLD/SLV sits at the 52.8th percentile of its trailing year. A catch-up trade
needs the ratio stretched; it is mid-range, so the 63d relative move is a
statement about the crash, not about today. Dismissed on that number. XME z10
+2.33, same thrust family as the miners.

**energy.** The most extreme 5d cluster in the tape: COP 100.0, XOP 99.2, XLE
98.4, DBC 96.8, OIH 94.4, USO 91.7, with crude +10.81% in five sessions and
XLE 1.74% off a 52w high while USO's 63d rank is 6.7. Dismissed: killed 08-10
(energy washout into CPI), 08-11 (XLE crude-beta residual, the [5,6)% band
watch W5 — today's 1d is -0.24%, no thrust at all), and 08-12 (the USO fade as
roll cost). The equity-vs-crude divergence form is the same crude beta measured
from the other end.

**dollar / FX.** DX-Y.NYB 100.01 with z10 -0.84 and a 21d rank of 26.2, 3.94%
off its 52w low. Not an extreme, and the 08-10 kill of the dollar-pullback cell
plus the live DX position mean the price state is not the way in. Appears only
as C4c's event leg.

**international.** EWZ is the tape's worst 5d rank at 2.4 (also 21d 12.7, 63d
4.0) — dismissed, **killed 2026-08-12** on two-episode concentration and a
sign that reverses at rank 1. FXI is the interesting one: 5d rank 19.0 and
-2.36% while EEM is +1.13% and the 21d rank is 84.1, i.e. a sharp break inside
an intact thrust. → **CHECK C8**. EWJ at a 52w high with a 0.42 volume ratio:
a quiet new high, no mechanism to trade in 10 sessions, not examined further.

**volatility.** Three things at once: VIX 14.55 (63d rank 16.3), UVXY at its
52w **low** and SVXY at its 52w **high**, and **VIX3M/VIX at +27.4%, the 98.4th
percentile of its trailing year**. That last number is the one nobody in this
repo has looked at. Every vol kill on the book so far (08-11 SVXY into CPI,
08-12 short SVXY across PPI, 08-12 the low-VIX leg of the skew cell) is
*event*-anchored or keyed on the VIX **level**; a term-structure percentile is a
different measure with a different mechanism (roll yield versus complacency).
→ **CHECK C1**. ^SKEW 136.54 at a 5d rank of 74.2, below the 95 its watch
entry (W7) needs, and both of W7's arming conditions fail. ^MOVE covered above.

**sectors.** The dispersion inside defensives is the standout: IHI 21d rank
**100.0** (+13.94%) and 12.73% below its 52w high, with ABT, BDX, MDT, SYK, BAX
all at 98-100 and XLV at a 52w high (21d rank 88.5, 63d rank 97.2) — against
utilities at the bottom of the entire tape (PNW 1.2, EIX 1.6, CNP 2.0, SRE 3.2,
AEP 3.6, D 3.6, DTE 3.6, ETR 4.0, XLU 7.5). → **CHECK C6** (fade the medtech
thrust) and **CHECK C7** (the pair). C7 carries a known hazard: long XLU was
killed on 08-07, again on 08-10 and again on 08-12, so the pair only survives if
the XLV leg does the work. XLF 63d rank 97.2 and SMH 63d rank 3.2 are both
dismissed — the semis-laggard cell was **killed 2026-08-12** as a bear-tape
selector with today outside its sample, and XLF at a 63d extreme with no event
is a momentum-continuation story the book's Sector BO already trades.

**repo-native state.** The dial's ma10-63d has gone **29.5 -> 72.2 in 21
sessions (+42.8)** while SPY sits 0.10% off a 52w high, and Low Absorption Ratio
is on at the 2nd percentile. 63 days in the 2016+ dial history share the
"delta21 >= +30 while SPY within 1% of its high" state. Every dial result in the
registry is about *sizing* the book; nobody has asked what the dial's **rate of
change** says about direction. → **CHECK C9**.

## 3. Seasonal and cycle cells

August, day 13, **midterm year (year%4 == 2)**. The seasonal board (as of
08-05) flags **zero** A/B setups and carries only regime context: midterm book
win 56.4% vs 64.9% all-years, and midterm fades on OVS, LT Trend ST OS and
Indices Oversold Bounce. Midterm is treated as a conditioner on every candidate
below rather than an idea of its own, which is also how the two surviving
midterm results in the repo are framed (the event sleeve's T2 and the DX cell).
Second-half-August seasonality was examined on 08-11 through the JH anchor and
killed. The board's put/call complacency line is stale (reading as of 08-04);
today's live P/C is the 44th percentile, fear OFF, so that cell is not live.

## 4. Watchlist verdicts (8 active, all with today's numbers)

| # | entry | verdict |
|---|---|---|
| W1 | TLT from the NFP close, long end at its 52w floor | **PASS**, trigger unchanged. Turns on at the first non-midterm NFP; the next print is 2026-09-04, still midterm, and 16 td out besides. The 08-10 tdom-control caveat still stands and is still owed before it trades. |
| W2 | Long LQD / short HYG at joint 52w extremes | **PASS**, unchanged at 4 declustered episodes. The joint state is live today (HYG 0.00% off its high, LQD 0.75% off its low) but it is the same cluster that began 07-22, so today is a mid-cluster entry and the count needs >= 8 across >= 3 non-2018 years. |
| W3 | Long SVXY overnight into the CPI print | **PASS on tradeability, re-measure owed.** CPI printed 2026-08-12, which is exactly the event the trigger says to re-measure after, but the next CPI is 09-11 (+20 td) and no legal horizon reaches it. Logged for the 09-10 run rather than spent today. |
| W4 | Long GLD on a miner-led thrust the metal has not joined | **PASS**, trigger not live. Needs GDX 5d rank >= 95 while GLD's is < 95; today GDX is 88.9 and GLD 84.9, so the required *divergence* is absent, and the live GDX leg to 08-17 fails the third condition too. |
| W5 | Long XLE on a crude one-day pop in the [5%,6%) band | **PASS**, trigger not live. USO's one-day move is **-0.24%**. Not a pop in any band. |
| W6 | Long TLT with the whole IG complex pinned at 52w lows | **PASS**, price rung ON, freshness leg FAILS. TLT 0.23% / IEF 0.86% / LQD 0.75% all inside the tight tolerances, but the trigger requires the first trigger day in >= 10 sessions and this episode began 2026-08-03, so today is deeper into the same run than the 08-12 read that already failed it. |
| W7 | Long SPY on a skew spike alone | **PASS**, trigger not live and both arming legs fail. ^SKEW's 5d rank is 74.2 against the 95 required; SPY is 0.10% below its high against the >1% required; 2026 is a midterm year. |
| W8 | Fade a crude thrust out of a deep base with a print inside the hold | **PASS**, still 4 post-2020 episodes against the 8 required, and today has no thrust to fade (USO 1d -0.24%). |

Nothing on the watchlist fires today. W3 is the only entry whose re-measure has
come due, and it cannot be traded inside a 10-td horizon.

---

## 5. Candidate slate selected from the map

Ten cells examined, **eight selected for a check** (the two calendar cells the
registry already swept, VIX-expiry week and the August pre-opex run, are
dismissed above on their own numbers rather than re-tested). Seven asset
classes, four axes, two event-anchored and six price-state-anchored, with two
candidates crossing both modes (C5, C9).

| id | candidate | class | axis | anchor |
|---|---|---|---|---|
| C1 | Short-vol carry at a 98.4th-percentile VIX3M/VIX term spread — SVXY, measured beta-neutral against SPY | volatility | interaction_cell | price state |
| C4 | The 10 sessions into Jackson Hole — long duration (TLT), with **4b** gold and **4c** the dollar as the alternate legs | rates / gold_miners / dollar_fx | event_fingerprint | event |
| C5 | The macro vacuum: from the month's last inflation print to the next scheduled 08:30 release, no macro event inside the hold | us_large / volatility | interaction_cell | event x tape state |
| C6 | Fade the medtech thrust — short IHI at a 21d rank of 100 out of a 12.7% drawdown | sectors | inversion | price state |
| C7 | Defensive dispersion — long XLU against short XLV at opposite 21d rank extremes | sectors | relative_value | price state |
| C8 | China's break inside an intact thrust — FXI against EEM | international | relative_value | price state |
| C9 | The dial thrust: SPY forward returns when the ma10-63d fragility dial rises >= 30 points in 21 sessions while SPY sits within 1% of its 52w high | us_large | interaction_cell | repo-native state x price state |
| C10 | The Dow's five-day break against its own 63-day leadership — DIA against SPY | us_large | relative_value | price state |

Negative-registry collisions declared up front, to be answered inside each
check rather than left for the red team:

- **C1 and C5 both touch SVXY**, whose event-anchored forms died on beta-neutral
  residuals and placebo ladders (08-11 into CPI, 08-12 across PPI), and whose
  *calendar*-gated carry form died on 2026-08-07. Every vol check today leads
  with the SPY residual and a placebo anchor ladder, and states which side of
  the 2018-02 -1x-to-0.5x leverage cut its sample sits on.
- **C7's long-XLU leg is dead in four expressions** (outright washout, the XLP
  pair, the SPY spread, the rates channel) and the registry says not to reopen
  it without a new mechanism. Dispersion against the *leading* defensive is the
  proposed new mechanism; if XLU turns out to be carrying the pair, that is a
  kill, and the XLV leg must be measured on its own.
- **C9 sits beside the entire dial-conditioning graveyard**, but every entry
  there is a *sizing* rule for the systematic book (throttles, caps, ramps,
  boosts). Whether the dial's rate of change says anything DIRECTIONAL has
  never been asked. The check must also state which dial vintage it used: the
  parquet is point-in-time only since 2026-07-02 and a recompute vintage before
  that, drifting up to ~7 points.
- **C6 and C10 both live near "conditional cells that underperform their own
  instrument's drift"** and near the AAPL-laggard kill; excess over own drift is
  the headline statistic in both, not the raw mean.
