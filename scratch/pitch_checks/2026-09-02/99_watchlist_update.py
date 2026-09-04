"""Post-publish watchlist maintenance for 2026-09-02.

Three jobs, in the order the skill states them:
  1. RE-ARM watchlist 4. Its stated arm fired on every leg today and the cell
     died anyway on four independent grounds. Left alone it reads ARMED again
     on the next thrust and burns a checker, which is exactly what the
     2026-09-01 flattener re-arm was written to prevent.
  2. Record that the parabolic-run route to paying watchlist 29's lag debt is
     now closed.
  3. Append today's three near-misses with the number each turned on, and
     stamp a dated verdict note on every entry read this morning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TODAY = "2026-09-02"
wl = load_watchlist()
ents = wl["entries"]
print("before: %d entries" % len(ents))

by_title = {e["title"]: e for e in ents}


def note(title_frag, text):
    hits = [e for e in ents if title_frag.lower() in e["title"].lower()]
    assert len(hits) == 1, (title_frag, [h["title"] for h in hits])
    hits[0]["note"] = "%s verdict: %s" % (TODAY, text)
    return hits[0]


# ---------------------------------------------------------------- 1. re-arm 4
e4 = note("crude one-day thrust in the 5 to 6 percent",
          "ARM FIRED AND THE CELL DIED. See the re-armed trigger below.")
e4["trigger"] = (
    "RE-ARMED 2026-09-02 after the old arm FIRED IN FULL and the cell died "
    "anyway. What happened: USO popped +5.460% into the [5,6)% band at 1.65 "
    "ATR with PPI (+5 td) and CPI (+6 td) both outside a three-session hold, "
    "so every stated leg cleared, and the parked numbers reproduced to the "
    "decimal (38 episodes, +1.251% mean, excess +1.105pp, 73.7% hit, sign p "
    "0.0025). It died on four independent grounds. (1) THE ATR LEG IS A FILTER "
    "THAT DOES NOT FILTER: >=1.50 ATR moves the excess from +1.105pp to "
    "+1.121pp, 1.6 basis points, while discarding 22 of 38 episodes whose own "
    "excess is +1.094pp. The condition that made this an arm rather than a "
    "state removes 58% of the sample and separates nothing. (2) DEFINITION "
    "FRAGILITY: the bucket ladder is an interior spike with [4,5) at -0.186pp "
    "and [6,7) at +0.499pp, and sliding the lower edge to 4.8% halves the "
    "result to +0.685pp. A per-session decomposition puts 76% of the three-day "
    "number on day three, with day four giving back -0.774%, so h=3 is a "
    "one-day spike rather than a hold. (3) CONCENTRATION: top three of sixteen "
    "armed episodes are 72% of the total, drop-best-3 leaves +0.292pp, and the "
    "only recent era is wrong-signed (2026 armed -0.451% on n=2, band -1.217% "
    "on n=4). (4) THE BETA RESIDUAL SURVIVES THE BAND: re-estimated crude beta "
    "0.506 (t 57.7), armed residual +0.506pp at a 62.5% hit and sign p 0.227, "
    "band residual +0.827pp at a 55.3% hit and sign p 0.314 - the registry's "
    "2026-08-11 closure intact. TURNS ON only when the crude-beta residual "
    "clears on its own terms: sign p <= 0.10 at a hit rate above 65% on at "
    "least 20 band episodes, AND the [4,5)% bucket ceasing to be negative "
    "(-0.186pp today), so the band is a shelf rather than a spike. The ATR leg "
    "is RETIRED from the arm; do not re-add it without fresh attribution. Two "
    "findings that do NOT need re-testing: an NFP inside the hold is NOT the "
    "CPI/PPI containment effect (+1.973% on n=7 against +1.088% on n=31, Welch "
    "t +0.64), and the CPI/PPI exclusion IS real (-0.550% in against +1.985% "
    "out, Welch t -2.51). Standing book overlap, verified again: 47 Overbot "
    "Vol Spike SHORT signals on USO >= +5% days at avgR +0.286, and on armed "
    "trigger days 9 of 10 energy positions the book held across the hold were "
    "SHORT at avgR +1.046. Dial max on any armed episode 80.9 against today's "
    "87.5."
)
e4["script"] = "scratch/pitch_checks/2026-09-02/c1b_energy_band_r2.py"
e4["added"] = TODAY
e4["expires"] = "2027-09-02"

# ------------------------------------------------------- 2. close 29's route
e29 = note("Short silver after the whole metals complex breaks",
           "PASS, and one of its two routes is now CLOSED. The three-name "
           "break was LIVE for the first time since 2026-08-28 (SLV -3.68%, "
           "GLD -2.86%, GDX -3.90%) but arm (a) needs a break of -4.00% or "
           "worse, so the depth bucket is unmet. The PARABOLIC-RUN route to "
           "paying the lag debt is now closed: gating the break on a miner "
           "21-day rank >= 90 pays +0.185% on 4-8 against the bare parent's "
           "+0.454% on 87-90 and the discards' +0.521%, i.e. -0.270pp at h=1 "
           "and -1.344pp at h=5, monotone as the gate tightens, and the gated "
           "lag profile is still lag=0 -0.297% / lag=1 +0.185% / lag=2 "
           "-0.420%. Today's -3.68% break also sits in the losing half: breaks "
           "worse than -3.0% pay -0.716% on 1-6. "
           "(a1_c3_slv_parabolic_break.py)")

# ------------------------------------- 3. dated verdict notes on the rest read
NOTES = {
    "Long TLT from the NFP close": "PASS. Still midterm; NFP is +2 td (2026-09-04). Parks to 2027-01.",
    "Credit-quality divergence": "PASS. Episode count still 4 against the 8 required. State live and tighter than yesterday - HYG -0.80% off its high, LQD AT its 252d low - and still uncountable.",
    "Long SVXY overnight into the CPI print": "PASS. CPI is 2026-09-11, +6 td, so the overnight entry is 5 sessions away.",
    "Long GLD on a miner-led thrust": "PASS on the opposite side of the trigger. GDX r5 18.3 against the >=95 leg: the miners BROKE (-3.90% on the day, -10.28% over five) rather than thrust, and GLD is -19.99% off its high against the added within-10% leg. Separately closed today: the outright long-miner form is a class effect with a dispersion ratio BELOW 1 at every horizon (0.89/0.82/0.78 across 14 names) and today's GDX 5-day rank of 9.1 is below the historical minimum of the cell's own support.",
    "Long TLT with the whole investment-grade complex": "PASS. TLT +1.02% above its 252d low against the <=0.5% rung; IEF +0.28% and LQD +0.00% both clear. Closer than yesterday's 1.44% and still the only failing leg.",
    "Long SPY on a skew spike alone": "PASS. ^SKEW r5 68.7 against the >=95 leg; midterm block stands.",
    "Fade a crude thrust out of a deep base": "PASS. USO r63 53.2 against the <=20 leg - crude is thrusting from mid-range, not out of a base.",
    "Long the medical-device thrust": "PASS. IHI r21 44.8 against the rank-100 rung.",
    "Long China's five-day break": "PASS. FXI r5 82.1 against the <=20 trigger.",
    "Long TLT on the NOVEMBER month-position": "PASS. Parks to trading days 4-12 of November 2026.",
    "Short SPY with the index at a 52w high": "PASS. SPY -2.07% off its high against <=0.5%; TLT +1.02% above its low against <=1%.",
    "Long SPY on a volatility pop inside an already-calm tape": "PASS, and note the pop legs DID fire (VIX +9.52%, SPY -0.69%) while the calm-tape leg failed hard: VIX's LEVEL sits at the 95.2nd percentile of its trailing 21 sessions and its 21-day return rank is 56.7, against a <=25 rung. A separate compressed-RANGE road to the same idea was swept and killed today at day-level Jaccard 0.008 from this entry, so this one is not affected either way.",
    "Long gold on an unconfirmed rate rise": "PASS. DX r21 43.7 against <=15, and the 21-session yield change is +11.0 bp against the +20 bp floor.",
    "Long technology against healthcare after a rotation gap": "PASS. One-day XLV-XLK gap +2.20pp against the >=+3.0pp rung.",
    "Short the dollar on a rate rise the currency does not confirm": "PASS. ^TNX r21 68.3 clears the >=65 leg for once, but DX r21 43.7 fails the <=20 leg.",
    "Short TLT after a big up day": "PASS. TLT 1d -0.41% against the >=+1.5% thrust rung.",
    "Short regional banks against the big-bank index": "PASS. KRE r5 15.1; the arm is an ex-crisis cost threshold no new episode can move.",
    "The duration-neutral curve position": "PASS. 252-session yield change +62.0 bp against the +78 bp arm. ^TNX closed AT its 252d max again, so the proximity leg is not the binding one - the magnitude floor is, exactly as the 2026-09-01 re-arm intended.",
    "The NARROW energy thrust cluster": "PASS. Count of the 11-name complex at z10 >= 2.0 is 0 against the [2,3] arm (highest SLB +0.73), even on a session crude popped 5.46%.",
    "Cross-sectional new-high breadth": "PASS. SPY -2.07% off its high against >2.0%; raw-21d fragility 62.8 against <=50.",
    "The sector washout into a 52-week high at h=7": "PASS, and the family arm moved further away today. XLI r5 4.0 clears the washout leg but is -7.39% off its high against within-5%; XLRE clears both rungs and is outside the nine-SPDR family the entry is written on. A neighbouring triple-rank-floor form was swept today and its nine-SPDR fixed-effect common excess is NEGATIVE at -0.381pp (t -1.18), which is independent evidence against the pooled-family hope this entry parks on. Dial 87.5 against an episode max of 68.6.",
    "The utilities washout with the long end hit": "PASS, rates leg wrong-signed a fifth straight session. XLU r21 10.3 against the <=5 leg, TLT r21 57.1 against the <25 rung.",
    "The bare dollar washout": "PASS. Parks to a non-midterm year; DX r21 43.7 is not a washout in any case.",
    "High yield printing a fresh 52-week high": "PASS. SPY -2.07% off its high against the >=2.0% leg; dial ma10 87.5 against the <50 requirement.",
    "The leader's deep correction": "PASS. SMH r63 0.4 so the state is live, but r5 23.8 against the <15 arm and the 23-ETF Cochran Q is unmoved.",
    "A pure rates repricing with zero credit stress": "PASS. HYG -0.80% off its 252d high against the <=0.25% rung, so the tight rung is not live even though LQD closed AT its 252d low.",
    "Long IEF for one session out of the Jackson Hole close": "PASS. Midterm, and the conference passed on 2026-08-28. Parks to 2027-09.",
    "The laggard that is STILL FALLING": "PASS. No holder of r21>=90 AND r63<=10 anywhere in the 29-name pool today.",
    "Long duration with the ten-year at a yield high and bond vol MID-RANGE": "PASS. ^MOVE trailing-252 LEVEL percentile 79.0 against the [40,50) band.",
    "The small-cap month-end OVERNIGHT in December": "PASS. Parks to a date.",
    "Energy closing at a fresh 52-week high on a session the INDEX fell": "PASS. The state IS live again (XLE closed AT its 252d max on a SPY -0.69% session) and the standing blocker is explicitly not a number and may not be waived. Second live instance in two sessions with no arm movement.",
}
for frag, text in NOTES.items():
    note(frag, text)

# ------------------------------------------------- 4. today's three near-misses
NEW = [
    {
        "added": TODAY,
        "title": "Short volatility into the payrolls print out of a dead 21-day range",
        "cell": "nfp x volatility, range-compression release at the print",
        "trigger": (
            "THE SESSION, and this one arms on a DATE rather than on a level, "
            "because the mechanism is real and today's entry was one session "
            "early. The paying cell enters MOC the session IMMEDIATELY BEFORE "
            "the print with the VIX 21-day high/low range (over its 21-day "
            "mean) in the bottom 15% of its trailing year, exits at the print "
            "close: SVXY +1.313% over 21 episodes, 15-6, t 2.539, sign p "
            "0.039, excess +1.190pp, bootstrap P(mean<=0) 0.006, worst "
            "-4.125%; ^VIX -2.143% on a 22.2% up-rate over 45 episodes; UVXY "
            "-2.992%. Gate attribution is genuine - the 157 DISCARDED payroll "
            "anchors pay +0.248% and the compression state ALONE on 586 "
            "non-payroll days pays -0.018%, so neither leg works without the "
            "other - and the placebo ladder SUPPORTS it, which is rare here: "
            "across k=-8..+8 the true anchor ranks 1 of 17 (neighbours k=-3 "
            "-0.780pp, k=-1 -0.268pp). Robust to the range definition "
            "(absolute range +1.264% on n=31; SPY realised vol +0.974% on "
            "n=38) and the gate correctly excludes 2018-01-31 at a range "
            "percentile of 85.7, so Volmageddon is out of sample by "
            "construction. WHY IT WAS NOT TAKEN 2026-09-02: NFP was three "
            "sessions after the signal close, so the pitch entry was k=-3 h=2, "
            "a DIFFERENT cell paying +0.511% at a 54.5% hit, sign p 0.416 and "
            "bootstrap 0.208, and the extra session it buys costs -0.657% at a "
            "50.0% hit with a -6.77% worst case. No horizon rescues k=-3. "
            "TURNS ON at the run whose entry session is the LAST SESSION "
            "BEFORE a payrolls print, provided the 21d-range / 21d-mean "
            "trailing-252 percentile closes at or below 15.0 on the signal "
            "day. The next such date is 2026-10-01 (entry) for the 2026-10-02 "
            "print, unless a rescheduled print lands sooner. THREE DEBTS THAT "
            "MUST BE DISCHARGED EVEN WHEN ARMED, none waivable. (1) The "
            "fragility dial: max ma10-63d on any of the 15 dial-covered gated "
            "anchors is 68.0 and only 1 of 15 is above 60, so a dial in the "
            "80s puts the live reading outside the population. (2) IT IS NOT "
            "PAYROLLS: the family permutation across NFP, CPI, PPI and FOMC "
            "gives family-wise P 0.2766, with CPI at -1.662% and FOMC at "
            "-1.745% showing the same VIX drop out of a dead range and only "
            "PPI inverting, so the event label is arbitrary and multiplicity "
            "applies across four kinds. (3) The tradeable sample is 13: SVXY "
            "changed leverage from -1x to -0.5x on 2018-02-28 and the "
            "post-break record is 8-5, +0.785%, sign p 0.290 (the pre-break "
            "+2.171% halving is consistent with leverage rather than a regime "
            "break, but the current instrument's own record is thin). The "
            "exact live cell - a September print in a midterm year with the "
            "gate on - has N=0."
        ),
        "script": "scratch/pitch_checks/2026-09-02/c9b_vol_nfp_round2.py",
        "source": "stand_down",
        "expires": "2027-09-02",
    },
    {
        "added": TODAY,
        "title": "The pooled sector triple rank floor, with the index-near-a-high gate taken OFF",
        "cell": "sectors, oversold-across-three-horizons, pooled family",
        "trigger": (
            "A REASON TO EXIST BESIDE THE BOOK, which is a harder arm than a "
            "number. The bare form is the strongest statistic produced on "
            "2026-09-02: a sector at a 5, 21 AND 63-day rank floor "
            "simultaneously pays +1.131% at h=10 pooled over the nine SPDRs "
            "across 493 episodes, edge +0.732pp over all days, fixed-effect "
            "common excess +0.773pp at t +3.06, Cochran Q homogeneous with "
            "I-squared 0.0% - a genuine family effect rather than a best-of-K "
            "draw. It is not pitchable for two reasons that no threshold "
            "cures. (1) IT IS THE BOOK: 153 systematic signals fired inside a "
            "[-1,+11] td window around these episodes (Overbot Vol Spike 58, "
            "Oversold Low Volume 37, LT Trend ST OS 26), so the pooled bare "
            "floor is generic oversold mean reversion the scanner already "
            "harvests, and repeating the scanner has zero value here. (2) The "
            "NOVELTY that would distinguish it - the index-near-a-high gate - "
            "SUBTRACTS 0.733pp, taking XLI from +1.646% on 65 episodes at t "
            "2.236 down to +0.913% on 11, and the nine-SPDR fixed-effect "
            "common excess of the GATED form is NEGATIVE at -0.381pp (t "
            "-1.18). TURNS ON if a future instance shows the near-high gate's "
            "h=10 attribution POSITIVE (today -0.733pp) - which would make the "
            "cell something other than the book's own oversold harvest - or if "
            "a conditioner is found that the scanner demonstrably does not "
            "trade, measured as under 20 book signals in the [-1,+11] td "
            "window against today's 153. Do NOT re-open this on the bare "
            "pooled statistic alone, however good it looks; that is the "
            "re-running-the-scanner failure and it is closed by design."
        ),
        "script": "scratch/pitch_checks/2026-09-02/c5c7_supplement.py",
        "source": "stand_down",
        "expires": "2027-03-02",
    },
]
ents.extend(NEW)

# ------------------------------------------------------------ 5. prune expired
today_ts = pd.Timestamp(TODAY)
keep, dropped = [], []
for e in ents:
    exp = e.get("expires")
    if exp and pd.Timestamp(exp) < today_ts:
        dropped.append(e["title"])
    else:
        keep.append(e)
wl["entries"] = keep
print("dropped %d expired: %s" % (len(dropped), dropped))
print("after: %d entries" % len(keep))
save_watchlist(wl)
print("saved -> data/pitch_watchlist.json")
