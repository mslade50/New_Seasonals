# Cell map — run 2026-08-17 (Mon), asof session 2026-08-17, next session 2026-08-18 (Tue)

Midterm year. `prices_fresh=True`, core bar 2026-08-17. 1213 cells scanned, 86 fired
(54 event / 32 price). BH crit p 0.0008, 2 passes. `dropped_by_cap` empty, so the map
below is the whole fired surface. Novelty: `delta_suppressed=false`, every fingerprint
`is_new`, no `repeat_blocked`.

Prior brief (2026-08-16) published: August expiry-week Monday (SPY/QQQ/IWM), the VIX
Monday inversion, the gold seasonal-vs-stretch pair, BTC down streak, ^NYA up streak.
Nothing there repeats below by fingerprint, but see the GOLD line: the state is the same
one I wrote about last night and it is dismissed on that basis rather than on its numbers.

## Session summary (what actually printed)

^GSPC -0.52% while ^VIX +6.60%, ^VVIX +7.36%, ^VIX3M +3.14%, ^SKEW +3.29%, ^MOVE +8.70%.
TLT -0.84% and closed at 81.35, exactly its trailing-252 minimum; IEF 0.73% and LQD 0.35%
off their own 52w lows; ^TNX 21d rank 86.5. Breadth 63.2% above the 200d against 72.4%
21 sessions ago. Grains bid: ZC=F +6.54% to a 52w high, ZS=F +3.60%, ZW=F +2.15%.

## Event lane

| trigger | verdict |
|---|---|
| `E:vix_expiry` k2, 18 subjects, expiry Wed 2026-08-19 | **DRILL** — the cluster of the night. QQQ n=319 +0.216% hit 58.9% t=2.37 record 188-129 sign p 0.0008, the only event cell to clear BH; SPY t=2.87, ^GSPC t=2.68, IWM t=2.54, all four era-stable and all pointing the same way. Anchor is today, so h1 is tomorrow. Needs conditioning before it is worth Scott's time: August, midterm, and above all whether it survives anchors where vol was BID into the expiry, which is tonight's actual state. Script 01. |
| `E:weekday_month` "Tuesdays in August", 18 subjects | **SKIP(degenerate)** — the bare day-of-week x month cell, and it is empty on every subject that matters: ^GSPC -0.013% t=-0.13 with a 58-59 record, SPY +0.005% t=0.05, QQQ -0.017%, IWM +0.006%. JPY=X is the only |t| over 2 (-0.114% t=-2.19, n=113) and its record is 50-63 with sign p 0.129, so the t is doing work the sign test will not confirm; era_stable false. Nothing here is more specific than "it is a Tuesday". |
| `E:seasonal_doy` Aug 18 (+/-2), 18 subjects | **DRILL, bonds only.** Equities are noise: SPY h1 all-years -0.087% on 13-13, midterm +0.329% on 3-3, n=6. The bond leg is the live one — TLT h5 18 up of 23, mean +0.672%, sign p 0.0053, and ^TNX h5 18 down of 25 at -0.93%, sign p 0.0378, two sides of the same claim. Directly relevant because TLT closed at a 52w low today. Also DX-Y h1 19 down of 26, sign p 0.0145. Script 04. |

Calendar inside the next five sessions, each with a verdict:

- **Wed 2026-08-19 VIX expiry (2 td)** — covered by script 01, the anchor is today.
- **Fri 2026-08-21 monthly opex (4 td)** — SKIP tonight. Anchored k1..k3, so the live
  anchors are 08-18 through 08-20, not today; publishing it now is the countdown
  re-telling the novelty rule bans. It earns its telling on 08-20.
- Jackson Hole 08-28 (9 td), NFP 09-04 (14 td), PPI 09-10, CPI 09-11, FOMC 09-16, quad
  witching 09-18 all sit outside the window. Calendar block only, no cell.
- No US releases printed today (`releases_today` empty), so the whole `P12` conditional
  family is inert tonight. Not a dismissal, there was nothing to condition on.

## Price lane

| trigger / subject | verdict |
|---|---|
| `P5`, `P5b`, `P6`, `P8` on **HE=F** (4 cells) | **SKIP(contract roll, not a market move).** The -14.39% session is a roll: 08-14 closed 95.40, 08-17 opened 81.60 and ranged 81.35-82.93. A 1.5-point intraday range does not produce a 14-point close-to-close move. Known failure mode for this ticker, verified on the bars tonight rather than assumed. Everything downstream is contaminated: the 2-ATR day, the bottom-5% 5d and 21d ranks, and the 200d cross are all the same artifact wearing four hats. |
| `P5:rank5` / `P7b` on **LE=F** | **SKIP(weak, and adjacent to the above).** Cattle is a genuine decline, 232.75 to 218.65 over four sessions with continuous intraday range, so it is not a roll. But the stats are empty: rank5 bottom h1 +0.014% t=0.14, down streak +0.031% t=0.32 on a 76-71 record. Nothing to say. |
| `P1` + `P1b` + `P5` + `P5b` + `P6` on **ZC=F** (5 cells) | **DRILL** — corn is the price lane's real event, +6.54% to a 52w high, 5d return at the 100th percentile of its year. Five triggers on one name is one story, not five. Base cells are middling (P5b rank21 n=417 +0.185% t=1.97 era-stable; P1b n=17 -0.049%), so the question is whether the momentum or the reversal cell governs when they fire together. Script 05. |
| `P4:z10_extreme` **GC=F** stretched up | **SKIP(published last night).** z10 +2.47, and the numbers are interesting in their own right, mean +0.001% against a 57.1% hit rate with sign p 0.009 and era_stable false, a clean mean/median divergence. But last night's item 3 was gold at z10 +2.18 in exactly this state. The fingerprint differs, the state does not, and re-telling it tomorrow with a bigger z is the failure the repetition rule exists to stop. |
| `P4:z10_extreme` **USDTRY=X** stretched up | **SKIP(mechanical).** BH-passing at n=581, hit 69.2%, t=3.15, and it means nothing: a managed depreciating currency drifts up almost every day, so "stretched up predicts up" is the crawl, not a state. Its unconditional drift is most of the +0.195%. |
| `P4` **SI=F** up, **SB=F** up, **USDMXN=X** down | **SKIP(weak).** SI=F +0.129% t=0.98; SB=F -0.003% t=-0.02; USDMXN n=84 +0.073% t=1.32. |
| `P5:rank5` top on **ZW=F**, **ZS=F**, **USDBRL=X** | **SKIP(weak / subsumed).** ZW=F t=1.46 on a 161-175 record, mean and sign disagreeing. ZS=F -0.059% t=-0.57. USDBRL -0.071% t=-0.64. The grain complex story is corn's, script 05. |
| `P5b:rank21` top on **EWJ**, **^GDAXI**, **SB=F** | **SKIP(degenerate).** ^GDAXI hit 54.6% with sign p 0.0435 but a mean of -0.003%, which is a hit rate describing nothing. EWJ 46.0% hit, -0.039%, t=-0.57. SB=F +0.046% t=0.42. |
| `P5b:rank21` bottom **CHFJPY=X** | **SKIP(mean/hit divergence, no magnitude).** 55.8% hit and sign p 0.0191 on a mean of +0.061%, t=0.8. The record is real and the move is not worth a sentence. |
| `P6:two_atr_day` up **ZS=F**; down **USDCNY=X** | **SKIP(weak).** ZS=F -0.081% t=-0.53. USDCNY +0.100% t=0.76 on a 48-47 record, sign p 0.92. |
| `P7:up_streak` **AUDJPY=X**, **USDBRL=X** | **SKIP(thin).** AUDJPY 56.1% hit, sign p 0.03, but mean +0.068% and t=1.67 — same no-magnitude problem. USDBRL -0.17% t=-1.01. |
| `P7b:down_streak` **^BVSP**, **^FCHI**, **^FTSE** | **DRILL(low priority).** ^BVSP is the best of the three, n=147 +0.39% t=1.85 hit 57.8% sign p 0.0346, era-stable, and it has an actual magnitude. ^FCHI hits 61.4% with sign p 0.0063 on a +0.139% mean and era_stable false, the divergence again. ^FTSE negative. Script 06 if the evening has room; ^BVSP only, and it dies unless the magnitude survives declustering. |

## States live tonight that no trigger fired on

The sweep is an enumerator, not the surface. Two things printed today that the trigger
definitions miss, and both are more relevant to tomorrow than most of what fired.

- **TLT closed exactly at its 252-day low while ^MOVE jumped 8.70%.** `P2` did not fire
  because its novelty filter wants the first 52w low in 30+ calendar days and TLT also
  closed at one on 2026-07-31, 12 sessions back. The filter is right about novelty and
  wrong about relevance: the state is live, it is the most macro-relevant thing on the
  tape, and bond vol spiking into it is not something the price lane has a trigger for at
  all. **DRILL**, script 03.
- **^VIX +6.60% on a -0.52% ^GSPC session.** `P10c` wants +10% so it did not fire, and
  `P9d` wants VIX up on an *up* day. Neither describes a vol bid this large against an
  index move this small, two sessions before a VIX expiry. **DRILL**, script 02.

## Plan

| script | cell |
|---|---|
| 01 | VIX-expiry-minus-2 for SPY/QQQ/IWM: era, concentration, local control, August, midterm, and anchors with vol bid into the expiry |
| 02 | VIX up 5%+ with ^GSPC down less than 1%: forward SPY and VIX, horizon scan |
| 03 | TLT at a 52w low with ^MOVE up 5%+: forward TLT and ^TNX |
| 04 | the late-August bond seasonal, drilled for era and concentration |
| 05 | corn's 52w high after a 2-ATR session: which of momentum or reversal governs |
| 06 | ^BVSP down streak, declustered, if the evening has room |
