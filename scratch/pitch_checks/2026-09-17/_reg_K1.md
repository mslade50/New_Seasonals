# K1 adjacent registry (quarter-end, month-end, FX, Japan)
1570:  idea.** The Japan washout passed every one of those: 42 episodes, +1.564% at
1571-  h=5, excess +1.460pp, 30-12, sign p 0.0040, 52x cost, era-stable both sides of
1572-  2018, top-2 episodes are LOSERS so concentration runs the right way, SPY above
1573-  its 200d on 66.7% of trigger episodes against a 71.6% base rate, EFA-hedged
1574-  residual +0.674% at a 71.4% hit with a LOYO floor of +0.433%, and a daily
1575-  EWJ/yen correlation of +0.020 that rules out the currency. The reference-class
1576-  permutation is a separate and stricter test and it is what killed it. Run it
1577-  BEFORE spending a round-3 development pass, not after.
1578-  (a4c_c11_class_null_ownvol.py)
1579-
1580-## Cells swept and empty (2026-08-21)
1581-
1582-- **The opex anchor crossed with every non-equity class, which closes the
--
1823:- **The month-end anchor on EQUITIES, first measurement in this repo.** See the
1824-  session-decomposition entry above for the kill. Additionally: month-matching
1825-  takes the raw +0.352% (t 2.647) to **+0.164% at t 1.25**, and dropping
1826-  November (the only month with t>2) to +0.065% at t 0.53; the live August x
1827-  midterm cell is **3-3 at -0.860%** (2002 -2.98, 2022 -4.47) against
1828-  non-midterm August +0.707%; 27% of the total is two 2008 episodes; and the
1829-  60-cell grid walked (15 offsets x 4 vehicles) gives Sidak familywise **0.877**
1830-  with SPY ME-5 ranking 14th. The IWM rescue fails the same way (raw +0.513% at
1831-  t 3.05, month-matched +0.301% at t 1.81, worst ME-1 session of any vehicle at
1832-  -11.76 bp). (a2_c2_spy_me5.py, a2b)
1833-- **The investment-grade complex pinned at 52-week lows, translated to IEF.**
1834-  The anchor leg is wrong-signed: IEF within 1.0% of its 52-week low predicts
1835-  IEF **-0.021% at a 49.2% hit** (excess -0.034pp), negative in every era and
--
2107:- **Month-end on FX, which completes the month-end anchor to three asset
2108-  classes (equities closed 2026-08-24, rates suspended 2026-08-24, FX closed
2109-  here).** The mechanism is the 4pm London fix rebalancing flow and it is
2110-  falsified in its own window: DXY's **ME-0 session pays -0.55 bp at a 45.6%
2111-  hit** against an all-days base of +0.10 bp, and **+3.57 bp (wrong sign) from
2112-  2020**; the window's total comes from ME-1/-3/-4, sessions the story does
2113-  not name. The pre-specified signed regression on relative US-vs-foreign
2114-  equity performance is slope -0.0157, **t -0.75, R-squared 0.0019**, with
2115-  non-monotone terciles. The ME-5 spike (+5.03 bp, 55.9%) is noise: rotation
2116-  null over the same 16-cell walk gives **P(max |t| >= 1.93) = 0.523**. Cost
2117-  4.11x on the index and **0.38x and wrong-signed on UUP**, the only vehicle
2118-  that trades as an ETF. August x midterm is N=7 at -0.156%.
2119-  (d1_c5_monthend_fx_r1.py, d1b)
--
2265:  **The month-end anchor is now closed on equities in the month-of-year
2266-  direction as well as the month-position one.**
2267-  (c9_month_of_year.py, c9b_gate_and_scan.py, c9c_scanned_session.py)
2268-- **XLRE out of a lagging base into a duration rally, the first real-estate
2269-  cell tested here.** Pitched rung is 2 episodes; gate value swings
2270-  +0.886 / -0.254 / **-2.422** / -0.640pp across h=3/5/7/10 and the
2271-  "TLT NOT rallying" complement beats the joint cell at h=5 and h=7. Era-matched
2272-  across ten sectors it ranks 2 of 10 then 4 of 10, family mean gate +0.476pp
2273-  with **nine of ten sectors positive, Cochran Q 6.07 on 9 df, I-squared 0%**,
2274-  random-date max-of-10 **P = 0.113**. Top-2 episodes are 61% of total
2275-  (2020-04-02 alone +15.32%); drop-best-3 is **4.1x** cost at h=3 and 2.6x at
2276-  h=5. The pooled family form is NEGATIVE at every horizon ex-2020, and deep
2277-  history agrees (IYR **0.7x cost** over 2000+, VNQ 2.0x). **Real estate is
--
2553:  The month-end anchor is now closed on five classes.
2554-  (a4_c11_sector_month_turn.py)
2555-- **The country-ETF thrust from inside a drawdown, the INVERSION of the closed
2556-  break-inside-an-intact-thrust family.** The drawdown clause subtracts
2557-  (+0.673% bare, +0.463% joint, **+0.713% complement**; pooled -0.138pp over 11
2558-  names) and today's own depth bucket is the worst of six (**(-15%,-10%]
2559-  -0.289% at a 50.0% hit**). This reproduces the 2026-08-10 silver finding on a
2560-  second asset class: **distance-from-high is a U-shaped noise carve, not a
2561-  conditioner**, and that now holds on metals and on country equity. Family
2562-  Cochran Q p 0.7879, I-squared 0.0%, common excess -0.230%.
2563-  (b1_c3_thrust_in_drawdown.py, b1b)
2564-- **The "V that turned", 21-day rank >= 90 with 63-day rank <= 10, pooled over
2565-  29 ETFs.** Bare momentum +0.476% (N=3,521, t 6.43); joint +0.370% (N=189,
--
2915:  and closed both ways. **The month-end anchor is now closed on six forms
2916-  across five asset classes.** The next genuinely new anchor is still the
2917-  September FOMC on 2026-09-16, which enters the horizon around 2026-09-02 and
2918-  is spoken for by the event sleeve's T1/T2 -- and the midterm T2 short is
2919-  gated on SPY's 21-day rank being under 50, which reads 91.3, so the sleeve's
2920-  own rule remains off. The practical consequence is unchanged and now six
2921-  sessions old: a price-state sweep is the only honest search mode. Today it
2922-  produced twenty-two candidates across ten asset classes and no survivor, and
2923-  the closest of them died on a lag profile that no other test in the battery
2924-  would have caught.
2925-
2926-## Method traps (2026-09-01, from a 12-candidate sweep that killed all 12)
2927-
--
4619:- **Long TLT against SPY from QE-12 into quarter-end rebalancing.** The final
4620-  sessions carry no quarter-end premium (**+0.218% against +0.233% at ordinary
4621-  month-ends**). The window's return sits on the post-FOMC (+39.8 bp) and
4622-  post-quad (+25.7 bp) sessions, and the rest sums to -0.30%. The ungated
4623-  window goes 48-48 at a -0.003% median, and the top two episodes are 76% of
4624-  the total. The 63-day stock-bond spread gate has no gradient (Spearman
4625-  -0.005, low tercile best). September-only runs 13-2 in 2002-2016 and 3-6
4626-  from 2017. (k1_c9_qe_ladder.py, _b, _c)
4627-- **Long HYG on a duration-driven flush.** The cell is 2022 plus 2026: 6-5,
4628-  drop-best-2 -0.062%, ex-2022 -0.123%. Its best episode (2022-03-14,
4629-  +1.775%) has a -0.07% residual after IEF and SPY. Long IEF on the same dates
4630-  loses -0.320%. (k1_c4_round1.py, _b, _c)
4631-- **The healthcare complex flush, outright and as a pair.** The outright does
