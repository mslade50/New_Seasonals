"""Fold 2026-09-04's near-misses into the watchlist and stamp today's verdicts.

Verdict text for the 36 incumbents comes from 00_surface_map.md section 4;
the numbers there were computed by 00_watchlist_readings.py this morning.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

WL = Path('data/pitch_watchlist.json')
ASOF = '2026-09-04'
D = 'scratch/pitch_checks/2026-09-04'

# Today's verdict line per incumbent, keyed by the leading words of its title.
VERDICTS = {
    "Long TLT from the NFP close": "2026-09-04 verdict: PASS. Today IS the payrolls print and 2026 is midterm, so the cell is blocked on the leg it was parked on. Parks to 2027-01. A NEW route was opened and closed today: conditioning the same post-NFP duration cell on the PRIOR print's surprise instead of the price state (see the entry added today).",
    "Credit-quality divergence": "2026-09-04 verdict: PASS on the episode count, unchanged at 3 declustered against the 8 required. LQD +0.27% above its 252d low, HYG -0.35% off its high.",
    "Long SVXY overnight into the CPI print": "2026-09-04 verdict: PASS. CPI is 2026-09-11, so the overnight entry is the 2026-09-10 close, four sessions out.",
    "Long GLD on a miner-led thrust": "2026-09-04 verdict: PASS, both legs failing. GDX r5 31.1 against >= 95; GLD -17.28% off its high against the added within-10% leg.",
    "Long XLE on a crude one-day thrust": "2026-09-04 verdict: PASS. USO's 1-day move is +0.67%, nowhere near the [5,6)% band.",
    "Long TLT with the whole investment-grade complex": "2026-09-04 verdict: PASS, same single failing leg for a fourth session. TLT +1.27% above its 252d low against the <= 0.5% rung; IEF +0.47% and LQD +0.27% both clear.",
    "Long SPY on a skew spike alone": "2026-09-04 verdict: PASS. ^SKEW r5 57.0 against >= 95. Note the tape flags ^SKEW stale; the reading is 2026-09-03 either way, and its trailing-252 LEVEL percentile is 49.2.",
    "Fade a crude thrust out of a deep base": "2026-09-04 verdict: PASS. USO r63 55.8 against the <= 20 deep-base leg; crude is thrusting from mid-range.",
    "Long the medical-device thrust": "2026-09-04 verdict: PASS. IHI r21 59.8 against the rank-100 rung.",
    "Long China's five-day break": "2026-09-04 verdict: PASS. FXI r5 51.4 against the <= 20 break leg.",
    "Long TLT on the NOVEMBER month-position": "2026-09-04 verdict: PASS. Parks to trading days 4-12 of November 2026.",
    "Short SPY with the index at a 52w high": "2026-09-04 verdict: PASS, both legs failing. SPY -0.61% off its high against <= 0.5%, missing by 11bp; TLT +1.27% above its low against <= 1%.",
    "Long SPY on a volatility pop": "2026-09-04 verdict: PASS, wrong-signed on the defining leg. ^VIX FELL -5.79% against a >= +5% pop, and its 21-day return rank is 29.9 against <= 25.",
    "Long gold on an unconfirmed rate rise": "2026-09-04 verdict: PASS on the dollar leg for a sixth session. DX r21 33.1 against <= 15. The yield leg stays emphatically live, with ^TNX closing at its 252-day maximum for a fourth session.",
    "Long technology against healthcare": "2026-09-04 verdict: PASS, and wrong-signed today. The one-day XLV minus XLK gap is -1.11pp against the >= +3.0pp rung.",
    "Short the dollar on a rate rise": "2026-09-04 verdict: PASS. ^TNX r21 78.9 clears >= 65 for a fourth session; DX r21 33.1 fails <= 20.",
    "Short TLT after a big up day": "2026-09-04 verdict: PASS. TLT 1-day +0.15% against the >= +1.5% thrust rung.",
    "Short regional banks against the big-bank index": "2026-09-04 verdict: PASS. The arm is an ex-crisis cost threshold that no single session moves.",
    "The duration-neutral curve position": "2026-09-04 verdict: PASS on the magnitude floor. The 252-session yield change is +55.1bp against the +78bp arm. ^TNX closed at its 252d max again, so proximity is still not the binding leg.",
    "The NARROW energy thrust cluster": "2026-09-04 verdict: PASS. Count of the 11-name complex at z10 >= 2.0 is 0 (max VLO +1.45) against the [2,3] arm, a third straight session at zero with XLE, XOP, CVX and VLO all at or beside 52-week highs.",
    "Cross-sectional new-high breadth": "2026-09-04 verdict: PASS, both legs failing. SPY -0.61% off its high against > 2.0%; raw-21d fragility 63.1 against <= 50.",
    "The sector washout into a 52-week high": "2026-09-04 verdict: PASS. XLI r5 10.4 clears the washout leg but XLI is -6.41% off its high against the within-5% clause.",
    "The utilities washout with the long end": "2026-09-04 verdict: PASS, rates leg wrong-signed a seventh straight session. XLU r21 39.4 against <= 5, TLT r21 43.0 against < 25.",
    "The bare dollar washout": "2026-09-04 verdict: PASS. Parks to a non-midterm year; DX r21 33.1 is not a washout in any case.",
    "High yield printing a fresh 52-week high": "2026-09-04 verdict: PASS, both legs failing. HYG -0.35% off its high against the <= 0.05% touch; dial ma10 87.8 against the < 50 requirement.",
    "The leader's deep correction": "2026-09-04 verdict: PASS, and the closest single-leg miss on the list for a second session. SMH r63 0.8 so the floor is live; r5 15.5 against the < 15 still-falling arm, missing by half a rank point.",
    "A pure rates repricing with zero credit stress": "2026-09-04 verdict: PASS. HYG -0.35% off its 252d high against the <= 0.25% rung.",
    "Long IEF for one session out of the Jackson Hole": "2026-09-04 verdict: PASS. Midterm-blocked and the anchor is six sessions gone.",
    "The laggard that is STILL FALLING": "2026-09-04 verdict: PASS. No holder of r21 >= 90 AND r63 <= 10 anywhere in the 218-name tape.",
    "Short silver after the whole metals complex": "2026-09-04 verdict: PASS. The complex bounced again (SLV +2.51%, GDX +3.95%, GLD +1.85%); the depth arm needs a break of -4.00% or worse.",
    "Long duration with the ten-year at a yield high and bond vol MID-RANGE": "2026-09-04 verdict: PASS. ^MOVE's trailing-252 LEVEL percentile is 62.7 against the [40,50) band.",
    "The small-cap month-end OVERNIGHT in December": "2026-09-04 verdict: PASS. Parks to December.",
    "Energy closing at a fresh 52-week high on a session the INDEX fell": "2026-09-04 verdict: PASS. XLE slipped to -0.74% off its 252d max so the at-a-high leg lapsed, and SPY rose +1.05% in any case.",
    "Long SVXY into a scheduled print out of a 21-day VIX range": "2026-09-04 verdict: PASS, second session in the dead half. Rel-range percentile 3.35, inside the (0,5] band that pays -0.096%. Today's OTHER volatility route, the holiday-closure anchor, was opened and killed outright (see the entry added today): ^VIX RISES +4.80% across a >= 3 calendar-day closure, so short vol into a long weekend is wrong-signed.",
    "The pooled sector triple rank floor": "2026-09-04 verdict: PASS, and the breadth has collapsed. Only XLI holds the 5, 21 and 63-day floor today, down from nine names, so the pooled form has nothing to trade.",
    "Long SPY into a scheduled print out of a dead 21-day VIX range": "2026-09-04 verdict: PASS. The dial is 87.8 against the arm, and the anchor was yesterday's session in any case; today is the print itself.",
}

NEW = [
    {
        "title": "The risk premium carried ACROSS an extended market closure, short the index and long volatility",
        "added": ASOF,
        "cell": "market-holiday closure x us_large / us_small / volatility",
        "trigger": ("A PRE-REGISTERED FORWARD TEST, because the number itself is already there and "
                    "the problem is where it came from. ^VIX rises +4.80% across a >= 3 calendar-day "
                    "closure over 180 gaps at 136-44, sign p 0.0000, against +2.19% across an ordinary "
                    "weekend and -0.28% on a plain overnight, so the effect is monotone in the extra "
                    "calendar day. Tradeable side 2018+: short SPY across the gap +0.251% at 37-22, "
                    "sign p 0.034, 12.5x cost; short IWM +0.424% at 40-19, sign p 0.0043, 14.1x. Labor "
                    "Day 2018+ is 8-for-8 on both, and ungating to all closures keeps most of the edge, "
                    "so Labor Day is not the carrier and the object is the closure. It is NOT shippable "
                    "as found: it surfaced inside the blockers run against C1 and C2 and is the exact "
                    "opposite of both, which is the corpse-recovered sign flip the 2026-08-07 registry "
                    "entry closed by name. TURNS ON when a rule written BEFORE the fact, naming vehicle, "
                    "side, entry and exit with no re-optimisation, has cleared TWO new >= 3 calendar-day "
                    "closures, AND the pooled 2018+ short-IWM overnight still means at least +0.424% at "
                    "14.1x cost once those two observations are added. The next closures are the 2026-11-26 "
                    "Thanksgiving and 2026-12-25 Christmas boundaries; verify each gap against the "
                    "master_prices index rather than assuming a calendar."),
        "script": f"{D}/a2b_gate_attrib_and_inversion.py",
        "source": "stand_down",
        "expires": "2027-09-04",
    },
    {
        "title": "Post-NFP duration after a MODERATE prior payroll miss, with a CPI inside the hold",
        "added": ASOF,
        "cell": "nfp x rates, conditioned on the PRIOR print's surprise",
        "trigger": ("THE BAND AND THE HOLD, and note this entry is narrower than the candidate that "
                    "produced it. Long TLT entered MOC on the print session, h=3, conditioned on the "
                    "prior NFP surprise: the (-100k,-50k] half pays +0.813% over 12 observations while "
                    "the <= -100k half pays -0.175% over 19 at 10-9, so the dose response runs backwards "
                    "and the shipped '<= -50k' cut was two different cells averaged. The whole gated edge "
                    "is also CPI-in-hold, at +0.670% over 13 episodes at a 76.9% hit against -0.127% over "
                    "18 at 44.4%. Today failed both: the prior print was -103k and the h=3 hold to "
                    "2026-09-10 contains PPI but not the 09-11 CPI. TURNS ON when a payrolls print has a "
                    "PRIOR surprise inside (-100k,-50k] AND a CPI lands inside the h=3 hold. Standing debt "
                    "that must be paid at the same time: the cell's full +6.43pp is more than accounted "
                    "for by 2021 alone at +7.51pp, so drop-2021 must be back above zero (it is currently "
                    "-0.043% on 25 at a 52% hit) before this is pitchable. Placebo ladder ranks the live "
                    "k=0 rung 5 of 17, which is a second standing debt. Conditioner is readable live: "
                    "macro_release_history.parquet is frozen at 2026-08-07 but NFP is monthly, so the "
                    "prior print is always in the file."),
        "script": f"{D}/b1b_c3_live_band_and_concentration.py",
        "source": "stand_down",
        "expires": "2027-09-04",
    },
    {
        "title": "Long SVXY at the first close AFTER an extended closure, on a clear calendar",
        "added": ASOF,
        "cell": "market-holiday closure x volatility, the k=+1 rung",
        "trigger": ("RUNWAY >= 4, plus a mechanism this entry does not yet have. The k=+1 rung was the "
                    "one thing that separated from both blockers aimed at it: a >= 3-day closure that is "
                    "not a print session with a clear calendar pays +0.710% at Welch t 2.34, and crossed "
                    "with runway it adds on top, with closure >= 3 and runway >= 4 paying +0.684% against "
                    "a plain weekend at the same runway paying -0.247%. It is NOT the pitched order: the "
                    "eve MOC ranks 15 of 17 on the placebo ladder and pays -0.795% at a 42.0% hit "
                    "post-2018, which is -9.9x an 8bp round trip, because the vol mark-down is taken "
                    "intraday on the eve and reverses across the gap. TURNS ON at the first close after a "
                    ">= 3 calendar-day closure with a runway of at least 4 sessions to the next scheduled "
                    "print, and only once the entry carries a STATED mechanism for why buying vol cheap "
                    "the session after a closure works when holding it through the closure loses. Note "
                    "2026-09-08 does NOT qualify: the runway from that close to PPI on 2026-09-10 is 2."),
        "script": f"{D}/a1b_svxy_gap_decomp.py",
        "source": "stand_down",
        "expires": "2027-09-04",
    },
]

wl = json.loads(WL.read_text(encoding='utf-8'))
entries = wl.get('entries', [])
hit = 0
for e in entries:
    for pre, note in VERDICTS.items():
        if e.get('title', '').startswith(pre):
            e['note'] = note
            hit += 1
            break
    else:
        print('NO VERDICT MATCHED:', e.get('title'))

titles = {e.get('title') for e in entries}
added = 0
for n in NEW:
    if n['title'] in titles:
        print('already present:', n['title']); continue
    entries.append(n); added += 1

wl['entries'] = entries
wl['asof'] = ASOF
WL.write_text(json.dumps(wl, indent=1), encoding='utf-8')
print(f'verdicts stamped {hit}/{len(entries)-added}, new entries {added}, total {len(entries)}')
