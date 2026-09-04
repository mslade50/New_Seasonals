# Crisis Alpha / Long-Vol Sleeve — Prototype Study

Run date: 2026-07-02. Track: `crisis-alpha`.
Code (all runnable, in this dir): `ca_prep.py` (data prep, writes ca_*.parquet),
`ca_overlays.py` (main prototype backtests), `ca_episodes.py` (episode attribution +
throttle-vs-hedge integration), `ca_robust.py` (LOEO / lag / hysteresis / t-stats),
`ca_final_stats.py` (summary stats), `ca_inspect.py` / `ca_inspect2.py` (data checks).
Saved series: `ca_sleeves.parquet` (daily overlay returns), `ca_prices.parquet`,
`ca_frag.parquet`, `ca_book_daily.parquet`, `ca_book_monthly.parquet`.

## Verdict in one paragraph

The skepticism in the brief is confirmed for puts and for anything held permanently:
always-on short-dated VIX exposure bleeds to zero (VXX-proxy CAGR **-52.9%/yr**
2011-2026) and an always-on 3M 5%-OTM SPY put burns **-7.0%/yr of NAV**. Tactically
gating by the live sizing dial (63d MA10 >= 55, top decile) rescues exactly one
structure: a small long position in 1x short-term VIX futures (VXX-proxy, built by
de-levering UVXY), in-market only 10% of days, which made **+$39k on $750k over 10
years at 5% NAV** (+0.52%/yr), paid in 7 of 11 gate episodes, survives dropping any
single episode (worst case +$22.6k ex-COVID), and is insensitive to entry lag and
hysteresis — but is statistically indistinguishable from zero (monthly t=+1.24,
p=0.22, N=119). Gated puts are at best breakeven under honest (skew + haircut)
pricing and flip negative on leave-one-episode-out; the in-repo prior art's "net
positive at all thresholds" result does not survive skew. The deeper negative
finding: the fragility dial is a pre-correction detector, not a crash detector — it
was OFF for Volmageddon (frag max 35.6), turned off 2020-03-02 *before* the COVID
crash proper, was off for the entire 2022 bear (frag max 4.9 Apr-Oct), and was on in
only **2 of the book's 12 worst months** since 2016. A frag-gated hedge therefore
cannot hedge the book's realized drawdowns; it monetizes fragility *transitions*.
Since the book's whole-curve maxDD since 2016 is only -$62k against +$2.35M PnL,
there is no crisis to insure. **Ship the pending sizing taper; do not adopt a
standing crisis-alpha sleeve.** The tactical VXX-proxy is defensible only as cheap
optionality (expected cost ~0), never as edge.

## 1. Prior art: `tests/backtest_put_hedge.py` (2026-02-20 session)

Design: 50/50 barbell of 40-delta (core) and 5-delta (tail) 3-month SPY puts,
**daily** delta-dollar rebalancing to a target of (fragility/100) x portfolio when
the raw (unsmoothed) 63d fragility score exceeds a threshold; priced with
Black-Scholes at flat VIX3M vol, no skew, no bid-ask; $100k base, ~10y window.
Documented results (notes.md): net P&L positive at every threshold — +$7.5k (thr 50)
to +$24.8k (thr 70) — at annualized premium spend of 15-32% of the portfolio, with
70 the "sweet spot". Its own header flags the biases: no skew (materially
underprices the 5-delta wing), no spread, flat vol.

My re-implementation with a linear skew (+4 vol pts per 10% OTM), 5% premium
haircut each way, and the *live sizing basis* (63d 10d-MA) instead of the raw score
shows those results do not survive: the gated 5%-OTM put nets only +$9.3k over 10y
at thr 55 on a 7.5x larger base ($750k), and is **negative at thr 50 (-$36.9k)**.
The prior art's positive net was mostly the free skew. Treat notes.md's put-hedge
table as superseded.

## 2. Data available / missing

In `data/master_prices.parquet` (adjusted daily OHLCV): UVXY (2011-10+, 2x until
2018-02-27 then 1.5x — de-levered daily to a 1x "VXX-proxy"), GLD (2004-11+),
TLT/IEF (2002-07+), DBC (2006+), TMF, SPY, ^VIX, ^VIX3M (2006-07+), ^IRX.
**Missing:** VXX/VIXY (proxied), SVIX/SVXY, DBMF/KMLM/CTA (no managed-futures ETF —
the sibling `trend-following` track built TSMOM from scratch and found it loses
-0.23%/mo in high-frag months, so that box is covered and negative), TAIL, BTAL,
options chains (puts are modeled, not market-priced). Fragility exists only from
2016-07 (`data/rd2_fragility.parquet`), so every gated backtest is a 10-year, ~11
episode sample. Requirement "gross performance 2003+" is unmeetable for a gated
sleeve; always-on baselines are shown from max availability instead.

## 3. Rules as implemented (`ca_overlays.py`)

- **Gate:** frag63 `.rolling(10).mean()` (the live sizing basis) >= 55 at close t
  (55 ~= p90 of the daily score = top decile), off below 50 (5-pt hysteresis).
  Position effective session t+1 (stage-next-open workflow), close-to-close returns.
- **Tactical long-vol:** w% of flat $750k NAV in VXX-proxy (UVXY daily return / 2.0
  pre-2018-02-28, / 1.5 after). Cost 10 bps/side on position changes. w = 2% and 5%;
  thresholds 50/55/60 tested.
- **Duration/gold:** TLT 20% NAV, IEF 20%, GLD 10%, tactical and always-on, 5 bps/side.
- **Put overlay:** when gate turns on, buy 3M (63 td) SPY put, strike 95% of spot,
  on 1.0x NAV notional; IV = VIX3M + 0.40 x OTM% (so 5% OTM = +2 vol pts, 15% = +6);
  r = ^IRX. 5% haircut on premium at every buy/sell/roll. Marked daily by BS; sold
  when gate goes off; rolled 5 td before expiry if still on. Put-spread variant:
  long 95% / short 85% (sells the richer wing).
- **Costs everywhere:** >= 5 bps/side ETFs, 10 bps leveraged ETP, 5% option premium
  haircut per side (>> 5 bps of notional).

## 4. Always-on baselines — the bleed, quantified

| Instrument (buy & hold, max window) | CAGR | Vol | Sharpe | MaxDD |
|---|---|---|---|---|
| VXX-proxy (1x ST VIX fut), 2011-10+ | **-52.9%** | 69% | -0.78 | -100% |
| UVXY raw, 2011-10+ | -80.1% | 118% | -0.81 | -100% |
| GLD, 2004-11+ | +10.3% | 18.2% | 0.54 | -45.6% |
| TLT, 2002-07+ | +3.7% | 14.3% | 0.20 | -48.7% |
| IEF, 2002-07+ | +3.6% | 6.8% | 0.30 | -24.4% |
| DBC, 2006-02+ | +1.6% | 19.3% | 0.09 | -76.4% |

Always-on 3M 5%-OTM put on 1x NAV, 2016-07+: **-6.95%/yr of NAV**. A constant 5%
NAV VXX-proxy over the same window: -2.17%/yr of NAV (-19.7% cumulative). Held
permanently, every convex instrument here costs multiples of what the fragility
hole is worth (55+ trades: ~171 trades diluted from ~+0.6R to +0.17R ~= a few $10k
per decade). Permanent long-vol is dead on arrival; GLD/TLT are just beta sleeves
(GLD +0.21 corr to book = diversifier, not crisis alpha; TLT failed exactly when
needed, 2022).

## 5. Tactical overlays, monthly eval 2016-08..2026-06 (N=119 months, all net of costs)

High-frag month = calendar-month mean of the daily gate basis >= 50, N=16 (same
definition as the sibling track). Sleeve returns in % of flat $750k.

| Overlay | ann ret | ann vol | corr book | hiFrag avg/mo (N=16) | hiFrag hit | calm avg/mo | total 10y |
|---|---|---|---|---|---|---|---|
| VXX-proxy 2% NAV thr55 | +0.21% | 0.5% | -0.14 | **+0.10%** | **69%** | +0.00% | +2.1% ($15.6k) |
| VXX-proxy 5% NAV thr55 | +0.52% | 1.3% | -0.14 | **+0.25%** | **69%** | +0.01% | +5.2% ($39.1k) |
| VXX-proxy 5% NAV thr50 | +0.62% | 1.5% | -0.15 | +0.26% | 63% | +0.02% | +6.1% |
| VXX-proxy 5% NAV thr60 | +0.53% | 1.3% | -0.14 | +0.25% | 50% | +0.01% | +5.2% |
| Put 5%OTM 1xNAV thr55 | +0.13% | 2.4% | -0.12 | +0.13% | 44% | -0.01% | +1.2% ($9.3k) |
| Put 5%OTM 1xNAV thr50 | -0.50% | 2.6% | -0.12 | -0.12% | 38% | -0.03% | **-4.9% (-$36.9k)** |
| PutSpread 95/85 thr55 | +0.15% | 1.6% | -0.13 | +0.14% | 44% | -0.01% | +1.5% |
| TLT 20% NAV thr55 | +0.08% | 1.0% | -0.08 | +0.06% | 50% | -0.00% | +0.8% |
| GLD 10% NAV thr55 | -0.04% | 0.7% | -0.01 | +0.00% | 50% | -0.00% | -0.4% |
| GLD 10% NAV always-on | +1.19% | 1.5% | +0.21 | +0.06% | 50% | +0.11% | +11.8% |

Only the tactical VIX-futures proxy has the right shape: positive in high-frag
months with a positive hit rate, ~zero drag in calm months (it is simply not in the
market — 287 of 2,443 days on, 11 round trips in 10 years), negative correlation to
the book. Tactical TLT/GLD are dead (duration didn't protect 2022-01; gold is
uncorrelated noise at this frequency). Puts have hiFrag hit rates below 50% —
premium burn during the long, non-crashing episodes (2021 Apr-Aug: -$21k; 2024 Q4:
-$16k) eats the crash payoffs.

Sleeve standalone dollar curves: VXX-proxy 5% total +$39.1k, maxDD -$19.8k. Put
thr55 total +$9.3k, maxDD **-$54.3k** (it spends most of the decade under water).

## 6. Episode anatomy — 11 gate episodes, and the 5 worst

All 11 gate-on episodes (>=55 on / <50 off, gaps <= 10 td merged). Book PnL = trades
*signaled* inside the window, flat $750k. Throttle = pending rec (1.0x to 0.5x
linearly over frag 50-60, non-OVS only).

| Episode | days | peak | SPY dd | Book $ | Book thr $ | VXXP 5% $ | Put55 $ |
|---|---|---|---|---|---|---|---|
| 2017-09-21..10-10 | 14 | 59 | -0% | -105 | +439 | -3,812 | -4,325 |
| 2018-09-26..10-19 | 18 | 70 | -7% | -3,163 | -1,582 | **+9,681** | +12,884 |
| 2019-12-24..01-08 | 10 | 55 | -1% | +6,391 | +6,391 | -437 | -2,477 |
| 2020-01-28..03-02 | 24 | 72 | -12% | -3,286 | -3,787 | **+16,476** | +19,978 |
| 2021-04-30..08-04 | 67 | 78 | -4% | +97,857 | +83,488 | -8,403 | **-21,284** |
| 2021-08-26..09-21 | 18 | 66 | -4% | +26,487 | +19,513 | +2,387 | +8,271 |
| 2021-11-17..22-01-27 | 49 | 95 | -10% | **-39,279** | -21,590 | **+9,942** | +13,672 |
| 2024-08-29..09-12 | 10 | 63 | -4% | -3,915 | -4,350 | +3,260 | +1,820 |
| 2024-10-10..12-27 | 55 | 69 | -4% | +54,380 | +47,825 | -5,891 | -15,610 |
| 2025-02-26..03-06 | 7 | 56 | -4% | +540 | -1,465 | +7,732 | +5,221 |
| 2026-02-18..03-25 | 26 | 81 | -6% | +58,172 | +38,643 | +8,559 | +5,348 |

5 worst by peak score: 2022-01 (95), 2026-02 (81), 2021-08 (78), 2020-02 (72),
2018-10 (70). Aggregate across those five: book baseline **+$110.3k** (the book is
NET POSITIVE in its own worst fragility episodes — the 55+ damage is avgR dilution,
not losses), throttled +$95.2k (throttle gives up $15.1k), throttled + VXX-proxy 5%
**+$131.4k**.

## 7. Robustness (`ca_robust.py`)

- **Leave-one-episode-out, VXX-proxy 5%:** full +$39.1k; dropping any single episode
  leaves +$22.6k (ex-COVID) to +$47.5k. Positive in 7/11 episodes. Not one-crash
  dependent.
- **LOEO, puts thr55:** +$9.3k flips to **-$3.6k / -$10.7k / -$4.3k** dropping
  2018-10 / 2020-02 / 2022-01 respectively. Put-spread same pattern. Three episodes
  carry everything, under already-optimistic modeled pricing. Reject.
- **Entry lag:** t+1 +$39.1k, t+2 +$39.5k, t+3 +$34.1k — no same-day-execution magic.
- **Hysteresis 0/5/10:** +$37.2k / +$39.1k / +$33.4k — insensitive.
- **Threshold:** 50/55/60 all positive for the vol sleeve (+$46k/+$39k/+$39k at 5%);
  puts flip sign at 50 — another fragility red flag for puts.
- **Significance:** monthly mean +0.044%/mo, **t=+1.24, p=0.22** (N=119); hi-frag
  months only: +0.25%/mo, t=+1.06, p=0.31 (N=16). NOT significant. Ten years and 11
  episodes cannot establish this edge; adopt-or-not must be decided on structural
  grounds (convexity timing), not expectancy.
- **Vol entry level:** gate-on VIX ranged 9.7-20.9 — the dial buys vol before it
  spikes, which is why the sleeve isn't systematically buying tops. This is the one
  structural point in its favor.

## 8. Integration: "size down" vs "size down + convex hedge"

Full window 2016-07+ (flat $750k, exit-dated realized PnL):

| Variant | total PnL | maxDD ($, realized curve) | monthly Sharpe | worst month |
|---|---|---|---|---|
| Book baseline | $2,349k | -$62.1k | 2.77 | -$32.9k |
| + pending throttle | $2,325k (-$23.9k) | -$59.9k | 2.80 | -$32.9k |
| Throttle + VXX-proxy 5% | **$2,364k** | -$59.9k | **2.87** | -$32.9k |
| Throttle + put thr55 | $2,337k | -$59.9k | 2.80 | -$32.9k |
| Baseline + VXX-proxy 5% | $2,391k | -$62.1k | 2.84 | -$32.9k |

What the overlay adds beyond de-risking: the throttle costs -$23.9k of positive
expectancy (frag>=50 trades still average +0.19R) and trims the worst episode
(2022-01: -$39.3k to -$21.6k); the vol overlay adds back +$39.1k with convex timing
and lifts 5-worst-episode PnL from +$95k (throttle only) to +$131k. But every delta
here is under 2% of the book's $2.35M PnL, the worst month doesn't move (see next
section), and the Sharpe gain (2.80 -> 2.87) is inside the noise of a 119-month
sample.

**The disqualifying finding.** Of the book's 12 worst months since 2016-07, the
gate was on in exactly 2 (2021-11: hedge +$9.7k against book -$14.5k; 2024-10:
+$2.4k). The other ten — 2024-04 (-4.2%), 2019-10 (-3.4%), 2026-06 (-2.6%), 2020-08,
2022-06, 2016-10, 2018-05, 2022-09, 2024-06, and the partial 2026-07 — all happened
at fragility 0-45, hedge flat. And the gate missed every *real* vol event outside
its pattern: Feb-2018 Volmageddon (frag max 35.6), the March-2020 crash proper (gate
exited 2020-03-02), the entire 2022 bear (frag max 4.9 from April to October). The
fragility dial detects a specific pre-correction signature; the book's losses and
the market's crashes mostly come from elsewhere. There is no gating signal in this
repo that turns long-vol into insurance for this book, and the book's own realized
curve (-$62k maxDD in a decade) doesn't need insurance.

## 9. Per-year sleeve returns (% of NAV, net)

| Year | VXXP 5% thr55 | Put thr55 | PutSprd thr55 | | Year | VXXP 5% | Put | PutSprd |
|---|---|---|---|---|---|---|---|---|
| 2016 | 0.00 | 0.00 | 0.00 | | 2022 | +1.46 | +1.58 | +1.46 |
| 2017 | -0.51 | -0.65 | -0.54 | | 2023 | 0.00 | 0.00 | 0.00 |
| 2018 | +1.29 | +1.72 | +1.35 | | 2024 | -0.36 | -2.05 | -1.46 |
| 2019 | +0.06 | -0.01 | 0.00 | | 2025 | +1.03 | +0.20 | +0.15 |
| 2020 | +2.07 | +3.60 | +2.23 | | 2026 | +1.14 | +1.39 | +1.10 |
| 2021 | -0.95 | -4.53 | -2.82 | | | | | |

(Zeros = gate never on that year. CAGR/vol/Sharpe/maxDD for sleeves and combined
book are in §5/§8; a 2003+ table is impossible for a 2016-gated overlay — stated
plainly rather than backfilled with a fake gate.)

## 10. Execution fit

- **Instrument:** VXX (1x ST VIX futures ETN) — not in `master_prices.parquet`;
  would need a backfill like the LEV3X names. Or trade UVXY (already in the cache
  and the LEV3X universe) at 2/3 size (5% NAV VXX-equivalent = ~3.3% NAV UVXY ~
  $25k). Both trade hundreds of millions to billions/day; capacity is a non-issue.
- **Orders:** ~1 round trip per quarter (11 in 10 years). Gate computable post-close
  from the risk-dashboard pipeline (same series that will drive the sizing taper);
  stage a next-open MOO order, exit the same way. Trivial addition to `daily_scan`
  or a two-line check in `order_staging`. No options infrastructure needed.
- **If puts were adopted (they should not be):** IBKR SPY options, 8 tickets/decade
  plus rolls, but the study's pricing is modeled — adopting would require live-chain
  validation the repo has no data for.
- **Data the repo lacks:** VXX/VIXY history, any options chains (IV surface), DBMF/
  KMLM (managed futures — negative anyway per the trend-following track).

## 11. Honest bias inventory

1. **Fragility is a current-vintage reconstruction** with composite calibration
   lookahead (edge weights from a full-sample event study — established caveat).
   The gate quality is therefore overstated by an unknowable amount. This is the
   single biggest caveat and it cuts against adopting even the VXX-proxy sleeve.
2. **Threshold selection is in-sample:** 55 was chosen knowing the established 55+
   finding. Mitigation: 50/60 give the same sign and similar totals for the vol
   sleeve (puts do NOT pass this check).
3. **VXX-proxy construction:** UVXY daily return / leverage ignores UVXY's higher
   expense ratio and daily-reset compounding vs a true 1x ETN — errors are bps/mo,
   direction ambiguous. UVXY data starts 2011-10; no VIX-ETP history covers 2008.
4. **Put pricing is modeled, not market:** linear skew, no term structure, no event
   premium. Real protection is richer precisely when the gate fires; put results
   are optimistic and they are still ~breakeven. (Prior art without skew was more
   optimistic still.)
5. **Exit-date PnL attribution:** book daily curve books PnL at exit, understating
   intramonth MTM drawdowns (the true COVID-March book path was worse than shown).
   Overlay-vs-book coincidence at monthly granularity is unaffected.
6. **Costs:** 10 bps/side ETP, 5 bps ETFs, 5% option haircuts; ignores option
   commissions (~1-2 bps notional, immaterial at this size).
7. **Survivorship:** none (index ETPs only). **Adjusted-price basis:** relative
   returns recomputed per run — safe under the repo's dividend-adjustment invariant.
8. **Sample:** 10 years, 11 episodes, 16 high-frag months. Nothing here can be
   significant; the sleeve decision is structural, not statistical.
9. **2026-07 is a partial month** (-$32.9k book) and is excluded from monthly evals.

## 12. Recommendation

**Do not build a crisis-alpha sleeve; ship the pending fragility sizing taper
(kill the 1.25x boost, 1.0x through 50, taper to 0.5x by 60) as the sole treatment
for the frag hole.** The book has no realized-curve crisis to insure (maxDD -$62k
in a decade against +$2.35M PnL), the frag dial misses the book's actual worst
months (2 of 12) and every true vol event outside its pattern (2018-02, 2020-03,
all of 2022), always-on convexity bleeds (-53%/yr VIX ETP, -7%/yr NAV puts), and
gated puts are LOEO-fragile under optimistic pricing — the in-repo prior art's
positive put-hedge result does not survive adding skew. The one structure that
earned a qualified pass is a tactical VXX-proxy (5% NAV VXX or ~3.3% NAV UVXY, on
at frag63-MA10 >= 55, off below 50, next-open orders, ~1 round trip/quarter):
+$39k/10y, LOEO-stable, ~zero calm-month drag, pays in the transitions the taper
also fires on (2018-10, 2020-02, 2022-01, 2026-02). If convex optionality is wanted
on top of the taper it is the only defensible form — but it is statistically
unproven (t=1.2) and sits on a lookahead-tainted gate, so treat it as an optional
$0-cost attachment, cap it at 5% NAV, and never let it justify keeping size on
elsewhere.

## Adversarial verification (2026-07-02, independent recompute)

Verifier script: `verify_crisis-alpha.py` (+ `verify_ca_sharpe.py`,
`verify_ca_sharpe2.py`), written from scratch against the raw parquets — no ca_*
code reused. Gate rebuilt independently (63d MA10, >=55 on / <50 off, position
effective t+1); VXX-proxy rebuilt from UVXY (/2.0 pre-2018-02-28, /1.5 after);
put model reimplemented (BS, IV = VIX3M + 0.40 x OTM%, 5% haircut per side, roll
T-5); throttle reimplemented (non-OVS, 1.0 -> 0.5 linear over frag 50-60, as-of
signal date, ffill limit 5).

**Reproduced exactly or within tight tolerance:**
- VXX-proxy B&H 2011-10+: CAGR **-52.8%/yr** (claim -52.9), vol 69%, Sharpe -0.75
  (claim -0.78), -100% total. CONFIRMED.
- Tactical VXXP 5% thr55: total **+$39,081 to the dollar**; 11 round trips;
  287 in-market days (11.4% of 2,513 — report says 10%, immaterial). CONFIRMED.
- Monthly stats: t=+1.24, p=0.22, N=119; hi-frag +0.254%/mo, 69% hit, N=16 —
  all exact. CONFIRMED.
- LOEO: 7/11 episodes paid; worst = drop COVID -> **+$22,642** (claim +$22,604).
  Same 11 episodes, same dates (my ends are 1 td earlier — off-day convention).
  CONFIRMED.
- Always-on put: **-7.18%/yr of NAV** (claim -6.95) — independent implementation,
  within modeling tolerance. CONFIRMED.
- Gated puts: thr55 total **+$9,381** (claim +$9.3k), thr50 **-$36,679** (claim
  -$36.9k). CONFIRMED.
- Throttle cost **-$23,934** (claim -$23.9k) — exact. Hedge add-back +$39.1k —
  exact. maxDD -$62,073 baseline / -$59,870 throttled — exact. Sharpe ladder
  2.77 -> 2.80 -> 2.86 (claim 2.87) — reproduces ONLY when the partial 2026-07
  month is included, which §11 claims is excluded from monthly evals; ex-partial
  the ladder is 2.87 -> 2.91 -> 2.98 (same increments). CONFIRMED with that
  inconsistency noted.
- Book realized curve: +$2,349,563, maxDD -$62,073 — exact. Volmageddon frag max
  35.6, gate exit 2020-03-02, 2022 Apr-Oct frag max 4.9 — all exact. CONFIRMED.

**Caveats found (none overturn conclusions):**
1. **"2 of the 12 worst months"** counts the partial 2026-07 month among the 12
   (inconsistent with §11). Excluding it, 2022-01 (-$9.0k, gate ON, hedge +$9.9k)
   enters the list and the count becomes **3 of 12**. Slightly more favorable to
   the hedge than reported; the structural conclusion (gate misses most book
   damage and every non-pattern vol event) is unchanged.
2. **Put LOEO attribution differs per episode** under my independent mark model:
   I get flips-to-negative dropping 2018-10 (-$3.7k), 2020-02 (-$21.3k) or
   2026-02 (-$1.0k), vs their 2018-10/2020-02/2022-01 triple. Totals match, the
   "three episodes carry everything" headline holds, but per-episode put numbers
   are model-dependent and should not be quoted as precise.
3. **Corr -0.14 is to book monthly $ PnL, not monthly R** as labeled: on the R
   basis I get -0.09. Small negative either way; label imprecise.
4. **5-worst-episode aggregate** reproduces as +$104.2k / +$89.0k / +$125.1k vs
   claimed +$110.3k / +$95.2k / +$131.4k — a ~$6k uniform offset from the
   episode-end boundary convention (their windows end 1 td later, capturing
   boundary-day signals). The deltas (throttle -$15k, hedge +$36k) match; the
   "book is net positive in its own worst episodes" conclusion holds.

**Verdict:** all decisive claims stand. The recommendation (taper only, no puts,
VXXP optional and unproven) is supported by the recomputation.
