"""Watchlist maintenance for 2026-09-03.

- entry 33 (nfp x volatility) FIRED and the cell died on its own dose
  response, so it is RE-ARMED in place on the band rather than removed
  (the 2026-09-02 watchlist-4 precedent for an arm that fires and fails).
- a new entry for the SPY direction leg of the same pre-print cross.
- every surviving entry gets today's verdict note so tomorrow's B1 starts
  from a cited number rather than a re-derivation.
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WL = ROOT / "data" / "pitch_watchlist.json"
TODAY = "2026-09-03"

wl = json.loads(WL.read_text(encoding="utf-8"))
entries = wl["entries"]
by_title = {e["title"]: e for e in entries}

# --- 1. today's verdict notes, keyed by title prefix ------------------------
NOTES = {
 "Long TLT from the NFP close to +3td":
   "2026-09-03 verdict: PASS. Still midterm. The print is TOMORROW (2026-09-04) "
   "and the cell stays blocked; parks to 2027-01. Note the sibling PRE-print "
   "anchor was tested outright today and is separately dead (TLT -0.090% over "
   "287 anchors, live k=-2 rung DEAD LAST of 17 placebo offsets), so no rates "
   "route into this print exists in either direction.",
 "Credit-quality divergence, long LQD against short HYG":
   "2026-09-03 verdict: PASS on the episode count. The tight rung was re-counted "
   "today at 17 days / 3 declustered episodes in all history (2018-06-20, "
   "2026-08-03, 2026-09-01), two of them today's own cluster, against the 8 "
   "required. LQD is +0.12% above its 252d low and HYG -0.79% off its high. The "
   "standing 'duration wearing a credit label' debt was re-measured and CONFIRMED: "
   "LQD-on-IEF residual -0.005 / -0.059 / -0.245pp at h=1/3/10.",
 "Long SVXY overnight into the CPI print":
   "2026-09-03 verdict: PASS. CPI is 2026-09-11, so the overnight entry is the "
   "2026-09-10 close, five sessions out. Read alongside the re-armed clear-calendar "
   "entry below: CPI 2026-09-11 carries a runway of 3 sessions and is the next "
   "qualifying anchor for that arm too.",
 "Long GLD on a miner-led thrust":
   "2026-09-03 verdict: PASS, both legs failing. GDX r5 21.4 against >= 95, and GLD "
   "-18.78% off its high against the added within-10% leg. The drawdown condition "
   "was re-confirmed on a THIRD independent cell today: the deep half (<= -10%, "
   "where today sits) is 0-for-3 at -1.667% (h=3) against +0.745% at a 77.8% hit "
   "for the shallow half.",
 "Long XLE on a crude one-day thrust":
   "2026-09-03 verdict: PASS. USO's 1-day move is +0.11%, nowhere near the [5,6)% "
   "band, and the re-armed beta-residual condition is untouched.",
 "Long TLT with the whole investment-grade complex pinned":
   "2026-09-03 verdict: PASS, and still the closest rates entry on one leg. TLT "
   "+1.12% above its 252d low against the <= 0.5% rung; IEF +0.36% and LQD +0.12% "
   "both clear. Same single failing leg as 2026-08-12 and 2026-09-02.",
 "Long SPY on a skew spike alone":
   "2026-09-03 verdict: PASS, and the entry needs a convention note. ^SKEW r5 is "
   "57.1 against the >= 95 leg. Separately established today: at 144.12 ^SKEW sits "
   "at the 49.6th percentile of its own TRAILING 252 days (trailing-year median "
   "144.18) and the 90.4th of full history, so this entry's percentile convention "
   "must be stated when it re-arms. SPY is now -1.64% off its high, which clears "
   "the first regime leg for the first time; the midterm leg still blocks, and the "
   "midterm intersection was re-measured today at -1.106% (h=5) / -2.207% (h=10).",
 "Fade a crude thrust out of a deep base":
   "2026-09-03 verdict: PASS. USO r63 49.6 against the <= 20 deep-base leg; crude is "
   "thrusting from mid-range for a second session.",
 "Long the medical-device thrust":
   "2026-09-03 verdict: PASS. IHI r21 61.9 against the rank-100 rung.",
 "Long China's five-day break":
   "2026-09-03 verdict: PASS. FXI r5 48.4 against the <= 20 break leg.",
 "Long TLT on the NOVEMBER month-position effect":
   "2026-09-03 verdict: PASS. Parks to trading days 4-12 of November 2026.",
 "Short SPY with the index at a 52w high while the long end":
   "2026-09-03 verdict: PASS, both legs failing. SPY -1.64% off its high against "
   "<= 0.5%; TLT +1.12% above its low against <= 1%.",
 "Long SPY on a volatility pop inside an already-calm tape":
   "2026-09-03 verdict: PASS, and wrong-signed on the pop leg. VIX fell -6.98% "
   "against a >= +5% rung, and its 21-day return rank is 32.5 against <= 25.",
 "Long gold on an unconfirmed rate rise":
   "2026-09-03 verdict: PASS on the dollar leg. DX r21 43.7 against <= 15. The yield "
   "leg is emphatically live (^TNX closed AT its 252-day maximum), so this remains a "
   "one-leg miss.",
 "Long technology against healthcare after a rotation gap":
   "2026-09-03 verdict: PASS. One-day XLV minus XLK gap +0.77pp against the >= +3.0pp "
   "rung.",
 "Short the dollar on a rate rise the currency does not confirm":
   "2026-09-03 verdict: PASS. ^TNX r21 83.7 clears >= 65 for a third session; DX r21 "
   "43.7 fails <= 20.",
 "Short TLT after a big up day":
   "2026-09-03 verdict: PASS. TLT 1-day +0.10% against the >= +1.5% thrust rung.",
 "Short regional banks against the big-bank index":
   "2026-09-03 verdict: PASS. The arm is an ex-crisis cost threshold no single "
   "session moves.",
 "The duration-neutral curve position":
   "2026-09-03 verdict: PASS on the magnitude floor. The 252-session yield change is "
   "+51.9 bp against the +78 bp arm, DOWN from +62.0 bp yesterday because the "
   "trailing reference bar rolled. ^TNX closed at its 252d max again, so the "
   "proximity leg is still not the binding one.",
 "The NARROW energy thrust cluster":
   "2026-09-03 verdict: PASS. Count of the 11-name complex at z10 >= 2.0 is 0 (max "
   "SLB +0.97) against the [2,3] arm, a second straight session at zero even with "
   "XLE, XOP, COP, CVX and VLO all closing at 52-week highs.",
 "Cross-sectional new-high breadth on a survivorship-free universe":
   "2026-09-03 verdict: PASS, both legs failing. SPY -1.64% off its high against "
   "> 2.0%; raw-21d fragility 61.4 against <= 50.",
 "The sector washout into a 52-week high at h=7":
   "2026-09-03 verdict: PASS. XLI r5 0.8 clears the washout leg but it is -7.36% off "
   "its high against within-5%, and every one of the twelve complex names is 6-25% "
   "below its own high, so the 'intact trend' clause fails complex-wide rather than "
   "narrowly. A neighbouring twelve-name COUNT form was swept today and died on gate "
   "attribution, which is independent evidence against the family hope here.",
 "The utilities washout with the long end hit ALONGSIDE it":
   "2026-09-03 verdict: PASS, rates leg wrong-signed a sixth straight session. XLU "
   "r21 18.3 against <= 5, TLT r21 45.6 against < 25. Note the live utility story is "
   "two California names (PCG -26.8%, EIX -25.9% over five) with XLU itself only "
   "-1.93%, so it is not the sector state this entry is written on.",
 "The bare dollar washout":
   "2026-09-03 verdict: PASS. Parks to a non-midterm year; DX r21 43.7 is not a "
   "washout in any case.",
 "High yield printing a fresh 52-week high":
   "2026-09-03 verdict: PASS, both legs failing. HYG -0.79% off its high against the "
   "<= 0.05% touch; dial ma10 87.9 against the < 50 requirement.",
 "The leader's deep correction":
   "2026-09-03 verdict: PASS. SMH r63 0.8 so the floor is live, but r5 31.0 against "
   "the < 15 still-falling arm.",
 "A pure rates repricing with zero credit stress":
   "2026-09-03 verdict: PASS. HYG -0.79% off its 252d high against the <= 0.25% rung, "
   "even though LQD closed +0.12% above its own low. Re-counted today: the tight rung "
   "is 3 declustered episodes in all history, two of them the live cluster.",
 "Long IEF for one session out of the Jackson Hole close":
   "2026-09-03 verdict: PASS. Midterm-blocked and the anchor is five sessions gone.",
 "The laggard that is STILL FALLING":
   "2026-09-03 verdict: PASS. No holder of r21 >= 90 AND r63 <= 10 anywhere in the "
   "218-name tape.",
 "Short silver after the whole metals complex breaks":
   "2026-09-03 verdict: PASS. The complex BOUNCED (SLV +1.99%, GDX +3.13%, NEM "
   "+2.06%, GLD +1.52%), and the depth arm needs a break of -4.00% or worse against "
   "2026-08-28's -3.68%.",
 "Long duration with the ten-year at a yield high and bond vol MID-RANGE":
   "2026-09-03 verdict: PASS, and further away. ^MOVE's trailing-252 LEVEL percentile "
   "is 83.9 against the [40,50) band, with a 5-day return rank of 93.3. That elevated "
   "reading was tested as its own candidate today and the mechanism inverted: a "
   "bond-vol bid predicts LOWER forward equity vol (corr -0.0506 over 5,876 days).",
 "The small-cap month-end OVERNIGHT in December":
   "2026-09-03 verdict: PASS. Parks to December.",
 "Energy closing at a fresh 52-week high on a session the INDEX fell":
   "2026-09-03 verdict: PASS. XLE closed at its 252d max for a third session, but SPY "
   "ROSE +0.44%, so the down-index leg fails. The at-a-high-into-an-event sibling was "
   "swept today and the placebo ladder killed it (XLE's live rung 8 of 17).",
 "The pooled sector triple rank floor":
   "2026-09-03 verdict: PASS, and one route is now CLOSED with a correction. Nine "
   "names hold the triple floor today (PCG, EIX, XLI, DOV, SNA, ITA, HON, AMAT, TJX). "
   "The twelve-name industrial COUNT form was swept as the 'reason to exist beside "
   "the book' and died on gate attribution rather than on crowding: the count fires "
   "without XLI already at its own floor on 0 days in 6,707 sessions. The book-overlap "
   "charge this entry rests on did NOT reproduce in that form (289 ledger rows inside "
   "the windows = 6.2% against a 6.5% calendar share, only 2 on a complex ticker), so "
   "the arm now needs a form whose gate is not a restatement of its own index.",
}

matched = set()
for e in entries:
    for prefix, note in NOTES.items():
        if e["title"].startswith(prefix):
            e["note"] = note
            matched.add(prefix)
            break
missing = set(NOTES) - matched
assert not missing, f"unmatched note prefixes: {missing}"

# --- 2. re-arm the entry that fired ----------------------------------------
OLD33 = "Short volatility into the payrolls print out of a dead 21-day range"
e33 = by_title[OLD33]
e33["added"] = TODAY
e33["title"] = ("Long SVXY into a scheduled print out of a 21-day VIX range in the "
                "(5,15] band, with a clear calendar behind it")
e33["cell"] = "scheduled print x volatility, compression release on a clear calendar"
e33["expires"] = "2027-09-03"
e33["script"] = "scratch/pitch_checks/2026-09-03/a9_c1_live_rung_verdict.py"
e33["trigger"] = (
 "THE BAND, not the ceiling. This entry RE-ARMS after its date arm fired in full on "
 "2026-09-03 and the cell died anyway, and it is reframed on both axes the morning "
 "settled. AXIS 1, the event: it is not payrolls, it is a CLEAR CALENDAR. PPI's median "
 "runway to the next scheduled print is 2 td (43.9% at <= 1) against NFP's 5 (87.5% at "
 ">= 3), and splitting on runway erases the inversion that produced the old "
 "multiplicity charge (gated PPI SVXY -2.150% -> -0.113%, short ^VIX -4.929% -> "
 "+0.374%), with the split reproducing on CPI and FOMC. The pooled clear-calendar cell "
 "(deduped by anchor date) pays SVXY +0.910% over 56 anchors at 38-17, sign p 0.005, "
 "and short ^VIX +1.975% over 114 at t 3.717, monotone in runway (<=1 -0.805, >=2 "
 "+0.625, >=3 +0.954, >=4 +0.900); inside that subset NFP's family-wise P is 0.6181, "
 "i.e. the event label does no work, which is the coherence signature rather than a "
 "charge. AXIS 2, and this is what killed it: THE COMPRESSION GATE IS BIMODAL. "
 "rel-range (0,5] pays -0.096% over 25 anchors at 13-11 (short ^VIX -0.046%), against "
 "+1.465% at an 82.4% hit for (5,10] and +2.034% at 78.6% for (10,15]. It holds in both "
 "eras (pre-2018 5-6, post-2018 8-5), and it is not the VIX level or the term "
 "structure: the 2x2 has (5,15] paying at both VIX-level buckets (+1.773 / +1.674) "
 "while (0,5] is dead at both (+0.054 / -0.259), and today's contango bucket [12,18)% "
 "is the cell's BEST at +0.804%. 2026-09-03 read 3.57. TURNS ON when the VIX 21-day "
 "relative-range percentile (21d max minus min over the 21d mean, trailing-252 rank) "
 "closes in (5.0, 15.0] on the k=-2 anchor of a scheduled print carrying >= 3 trading "
 "sessions of clear calendar to the next one. That arm: SVXY n=31, 25-6, +1.722%, t "
 "4.943, sign p 0.0004, bootstrap P(mean<=0) 0.0000; short ^VIX +4.181%, t 4.552. Next "
 "qualifying anchors: CPI 2026-09-11 (runway 3), FOMC 2026-09-16 (runway 12), NFP "
 "2026-10-02 (runway 7); PPI 2026-09-10 is disqualified at runway 1. THREE THINGS "
 "ALREADY SETTLED, do not re-run them. (1) The fragility dial is a headwind and not a "
 "kill: corr(dial, next-session long SVXY) is -0.0486 at t -2.41 over 2,458 days (LOYO "
 "-0.043..-0.060), but on a benign tape matching this state the dial adds nothing "
 "(+0.057% below 40, -0.019% at 40-70, +0.056% at 70+), and the endogeneity defence is "
 "FALSE (corr(gate, dial) -0.100, mean dial 19.0 ON vs 25.2 OFF, Jaccard 0.167 with the "
 "production VRC signal). (2) The vehicle is SVXY, not short UVXY: risk-adjusted per "
 "session 0.3771 vs 0.3652 at corr 0.976, per 1 ATR of risk +0.331 vs +0.246, and "
 "UVXY's borrow is a soft cost that can fail while its left tail is unbounded (worst "
 "-12.63% vs -4.85%). VXX, VIXY, SVIX and UVIX are NOT in master_prices. (3) "
 "Development is done: h=1 only (short-^VIX edge +2.184pp at h=1, +1.593 at h=2, +0.275 "
 "at h=3, negative beyond); MOC beats every close-anchored limit as a whole variant "
 "(+0.910% vs +0.772 / +0.756 / +0.531 / +0.511 at 0.10 / 0.25 / 0.40 / 0.60 ATR); "
 "time-only, since every target and stop ties or loses; losers are 17 of 114 anchors at "
 "a mean of -1.864%, worst 2025-07-30 at SVXY -4.13% on ^VIX +21.89%. ONE DEBT THAT "
 "SURVIVES THE ARM: the trade is substantially levered equity beta on the print session "
 "(corr(SPY h=1, SVXY h=1) +0.626 gated, beta 1.75, R-squared 0.392), so it can never "
 "be composed beside a long-SPY idea and the write-up owes the residual."
)
e33["note"] = ("2026-09-03: THE DATE ARM FIRED AND THE CELL DIED. Re-armed on the band "
               "above. Today's rel-range percentile 3.57 sits in the flat (0,5] half.")

# --- 3. the new entry ------------------------------------------------------
entries.append({
 "added": TODAY,
 "title": "Long SPY into a scheduled print out of a dead 21-day VIX range",
 "cell": "us_large x volatility, the direction leg of the pre-print compression cross",
 "trigger": (
  "THE FRAGILITY DIAL, and secondarily a cost floor. The 2026-08-07 registry sweep "
  "covered POST-NFP equity direction and found it empty; the PRE-print session out of a "
  "dead range had never been measured and it is real at h=1: +0.236% over 45 episodes at "
  "28-17, sign p 0.0676, an excess of +0.197pp over SPY's own drift and +0.218pp over a "
  "trading-day-of-month matched control, bootstrap P(mean<=0) 0.051, 11.8x cost. Gate "
  "attribution is positive (+0.183pp: gated +0.236% against gate-off +0.033%, and the "
  "rel>15 complement -0.015%), it is era-stable (+0.185% pre-2018, +0.363% after), it is "
  "NOT midterm-dead (+0.277% on 11), it is not a bull-tape selector at h=1 (below-200d "
  "gated pays +0.290%), and 50 of 54 neighbouring specifications are positive. The "
  "horizon scan puts the entire return on the print session (h=3 edge -0.113pp), which "
  "is the right shape for the mechanism. TURNS ON at the first pre-print session with "
  "the rel-range percentile <= 15 AND the fragility dial's ma10(63d) BELOW 50, where the "
  "gated cell pays +0.402% (n=11, dial < 30) and +0.237% (n=3, dial 30-50); on "
  "2026-09-03 the dial read 87.9, the 95.4th percentile of its own series, and ZERO of "
  "the 15 gated anchors carrying a dial reading sit above 70, with the nearest available "
  "read (all payroll anchors at dial >= 70) at -0.331% on a 25% hit. SECOND BAR, both "
  "must clear: drop-best-3 has to reach 5x cost, and it is at 4.9x (+0.099% against a "
  "2 bp round trip) with 61% of a +10.63pp total in three episodes and the cell carried "
  "by 2009 (+4.75 over 4) and 2023 (+3.89 over 3). TWO CAVEATS NO TRIGGER CURES. (a) It "
  "is not payroll-specific: the identical construction pays MORE into FOMC (+0.366%, "
  "n=28, 64.3%, sign p 0.0925) and is negative into PPI (-0.124%), so the event is a "
  "1-of-4 pick and the 40-cell grid's Sidak bound is p 0.080; the honest form is "
  "probably the pooled clear-calendar version, which should be derived rather than "
  "assumed. (b) This entry and the SVXY entry above are ONE POSITION: corr(SPY h=1, SVXY "
  "h=1) is +0.626 on the gated set at beta 1.75, so they may never be composed together, "
  "and if both ever arm the vehicle question is which expresses the view better rather "
  "than which two ideas to ship. One neighbour note not to lean on: in the pitched "
  "definition the [0,5) rel-range bucket is the weakest of the three gated buckets "
  "(+0.090%, n=16, 9-7) against [10,15)'s +0.454%, but that inversion reverses under "
  "win=21/lb=504 and both absolute-range specs, so it is not load-bearing here the way "
  "it is for the SVXY entry."
 ),
 "script": "scratch/pitch_checks/2026-09-03/b5_c12_round2.py",
 "source": "near_miss",
 "expires": "2027-09-03",
 "note": None,
})

# --- 4. prune -------------------------------------------------------------
today = date.fromisoformat(TODAY)
live, expired = [], list(wl.get("expired") or [])
for e in entries:
    if date.fromisoformat(e["expires"]) < today:
        expired.append(e)
    else:
        live.append(e)
wl["entries"] = live
wl["expired"] = expired
wl["generated"] = TODAY

WL.write_text(json.dumps(wl, indent=1) + "\n", encoding="utf-8")
print(f"watchlist: {len(live)} active, {len(expired)} expired")
print("notes refreshed:", len(matched))
