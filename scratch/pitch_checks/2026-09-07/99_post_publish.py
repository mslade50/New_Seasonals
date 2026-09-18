"""Post-publish bookkeeping for the 2026-09-07 stand-down: watchlist adds and
the four entries whose status materially changed today."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

wl = load_watchlist()
entries = wl["entries"]


def find(sub):
    for e in entries:
        if sub.lower() in e.get("title", "").lower():
            return e
    raise SystemExit(f"watchlist entry not found: {sub}")


find("Long TLT with the whole investment-grade complex")["note"] = (
    "2026-09-07 verdict: the FRESHNESS leg CLEARED for the first time since parking "
    "(last tight-rung fire 2026-08-18 = 13 sessions, against the >= 10 arm), and IEF "
    "(+0.44%) and LQD (+0.25%) both clear. Only the TLT price rung is missing: it must "
    "CLOSE at or below 81.44, which is -0.93% / 1.26 ATR from 82.21. The cell reproduced "
    "at +0.385pp excess, 83.3% hit over 18 episodes, sign p 0.0038, 13.4x cost, zero book "
    "overlap. NEW STANDING CONSTRAINT: do NOT express it as a resting limit at that level. "
    "It is a lag-1 cell whose entire edge is session +2 (+0.402% at 83.3%); the session a "
    "limit fill adds pays -0.302% at a 38.9% hit, every limit variant measured is negative "
    "or inside cost, and getting filled selects against you (the 7 fills where the state "
    "actually armed pay -0.564% at D+1 while the 4 that reversed out pay +0.957%, and the "
    "reversal is unselectable at order time). Two further defects found while testing that: "
    "holding the TLT rung at 0.5%, the IEF/LQD rung is a knife edge with BOTH neighbours "
    "wrong-signed (0.5% -0.173pp, 1.0% +0.385pp, 1.5% -0.073pp); and the IEF leg's apparent "
    "contribution is an anchor swap, not filtering - at day level it deletes 4 of 80 days "
    "and moves the mean +0.025pp, while at episode level it moves it +0.321pp of the "
    "+0.385pp headline purely by re-anchoring the Sept-2022 episode from 2022-09-06 (-1.03%) "
    "onto 2022-09-19 (+1.68%). LQD, by contrast, does filter. "
    "(06_tlt_floor_limit.py, 06_tlt_floor_limit_dev.py)")

find("duration-neutral curve position")["note"] = (
    "2026-09-07 verdict: PASS, and the entry now has TWO binding legs rather than one. The "
    "^TNX trailing-252 maximum touch LAPSED on the 09-04 bar (4.784 against a 4.796 max set "
    "2026-09-01), so proximity binds alongside the magnitude arm, which needs a 4.956 close "
    "(+17.2 bp). The 252-session yield change is +60.8 bp against the >= 78 bp arm.")

find("Long SVXY into a scheduled print out of a 21-day VIX range")["note"] = (
    "2026-09-07 verdict: PASS, 0.63 percentile points below the band's lower edge and rising "
    "three sessions running (3.57 -> 3.97 -> 4.37). Today is not an anchor; 2026-09-08 is the "
    "PPI k=-2 anchor at runway 1, which the entry disqualifies. Next qualifying anchor is the "
    "CPI k=-2 session 2026-09-09 (runway 3), then FOMC k=-2 on 2026-09-14 (runway 12). Note a "
    "denominator correction: the VIX 21d relative-range percentile recomputes to 2.4 on a "
    "trailing-252 basis and 5.0 expanding, against the 1st percentile the risk block displays.")

find("Credit-quality divergence")["note"] = (
    "2026-09-07 verdict: PASS. CORRECTION to the entry's own text and to the 09-04 note: at "
    "the stated tolerances the declustered episode count is FIVE, not the 3 or 4 claimed - "
    "2018-06-11, 2018-07-20, 2018-09-07, 2026-08-04 and 2026-09-02, because the 24-day 2026 "
    "cluster has split in two. Still 1 distinct year ex-2018 against the >= 3 arm.")

entries.append({
    "added": "2026-09-07",
    "title": "Long SPY against short IWM when the fragility dial sits in its 56-70 band",
    "cell": "fragility dial x cross-section (large vs small) - the dial's FIRST cross-sectional content",
    "trigger": (
        "THE DIAL FALLING BACK INTO THE BAND, and note this entry is the OPPOSITE of the "
        "reading that generated it. Everything this repo has ever tested the dial for is "
        "book-wide DIRECTION and SIZING and all of it failed (aggregate PIT t -0.23, the "
        "book-wide throttle registry-dead). Nobody had tested it CROSS-SECTIONALLY. The pair "
        "is real: h=5 pays +0.517% over 71 episodes at 47-24, sign p 0.0043, bootstrap 0.0054, "
        "against an all-days -0.020% and a local +/-126td -0.061%; h=10 +0.998% on 26-13. It "
        "survives every overlap form (gap=21 +0.502%, gap=63 +0.596%, RUN-1st 9-4, YEAR-mean "
        "8-0 at sign p 0.0039), is LOYO-stable (+0.463% to +0.577%, all p <= 0.020), monotone "
        "in the threshold (q80 +0.376 -> q95 +0.556), and its top-3 episodes are MINUS 14% of "
        "total, so the winners are broad. Three attacks failed to dent it: no cheap substitute "
        "reproduces it (best is days-since-a-5%-drawdown at +0.377% on 43-24 with Jaccard 0.20 "
        "against the dial; the dial-ON/substitute-OFF residual is +0.502% at 31-16 against "
        "sub-ON/dial-OFF +0.230% at 26-21), it is NOT the static large-over-small tilt "
        "(unconditional dial-era drift +0.075%, 14.6% of the total, and conditional beats "
        "unconditional in all 8 years with episodes), and it is vintage-robust (drop-2021 "
        "+0.463% at 34-15; stressing the cut +/-3.5 and +/-7 dial points keeps every cell "
        "positive at +0.409 to +0.753). WHAT KILLED IT is LOCATION: the entire edge lives in "
        "[56,70) - 49 episodes, +0.594%, 35-14, sign p 0.0019 - while [70,80) is 14 episodes "
        "at +0.071% on a 6-8 record, and regressing the pair return on the dial reading gives "
        "a slope of -0.0053pp per point (t -0.33, R2 0.002), fitting +0.393% at today's 87.96. "
        "The [80,+) bucket is not independent support: four clustered Dec-2021/Jan-2022 dates "
        "plus four inside the currently running episode, which has already paid +0.02% twice "
        "(2026-08-27 and 2026-09-03). TURNS ON when the 10d-MA of the 63d dial closes back "
        "INSIDE [56, 70), which from 87.96 is a fall of at least 18 points. Deduct the Event "
        "Sleeve wash when it does: T2 shorts SPY $75k of the fixed $750k book, cancelling "
        "about 27% of a 30 bps-risk SPY leg for any session a T2 window overlaps."),
    "script": "scratch/pitch_checks/2026-09-07/10_dial_spy_iwm.py",
    "source": "near_miss",
    "expires": "2027-09-07",
    "note": "2026-09-07: parked the day it was found. Dial 87.96 = 99.15th percentile, 18+ points above the arm.",
})

entries.append({
    "added": "2026-09-07",
    "title": "Long ITA at a 21-day rank floor while the index sits near its high",
    "cell": "subsectors, washout x trend-intact - the class the nine-SPDR family does not contain",
    "trigger": (
        "A REFERENCE CLASS THAT WAS NEVER RUN, which is a harder arm than a number and is "
        "deliberately so. The cell looks strong: ITA at a 21d rank <= 10 with SPY within 2% of "
        "its 252d high pays +1.223% at h=10 over 29 episodes with bootstrap P(mean<=0) 0.010 "
        "and an edge of +0.659pp, and +0.543% at h=5 over 43 episodes (26-17, bootstrap 0.071, "
        "edge +0.261pp). Unlike an overlap artefact it gets STRONGER under stricter "
        "declustering - gap=63 h=10 +2.343% (12-3, sign p 0.018), gap=126 +2.110% (10-2, "
        "p 0.019), RUN-1st over 44 runs +1.254% (30-14, p 0.011), YEAR-mean 9-1 at sign p "
        "0.0107 at BOTH horizons - across 10 distinct years 2013-2026 with only 2021 negative, "
        "worst episode -4.94%, at 24.5x cost outright. Today is the deepest reading on the "
        "whole tape: ITA r21 2.8, -9.75% over 21 sessions, -10.90% off its 52w high. WHY IT IS "
        "PARKED: it is one subsector plucked from a scan whose PARENT FAMILY IS EMPTY - the "
        "same rule pooled over the nine SPDRs with sector fixed effects is negative at all five "
        "horizons against own drift (-0.074 to -0.305pp) and against SPY (-0.015 to -0.162pp), "
        "and XLI's own cell is -0.002% at h=5 with XLI-minus-SPY 15-15. That is exactly the "
        "shape the house has killed four times. TURNS ON when the identical rule run across a "
        "13-name subsector reference class (ITA IHI IBB XBI ITB XHB XRT XME XOP OIH KRE SMH "
        "IYR) puts ITA's excess outside a max-of-13 permutation at P <= 0.10. Two further debts "
        "if it ever arms: the beta-neutral form (ITA minus SPY) is only 3.8x cost at h=10 and "
        "2.8x at h=5, below the 3x bar, so the outright long is a directional beta bet with the "
        "index at its high - and the index-near-a-high gate starts from a negative prior in "
        "this repo, confirmed three times, including on this very cell (the gate-OFF washout is "
        "strongly positive at +0.222pp on N=1879). Defense also carries an obvious 2013-2019 "
        "versus 2022+ rearmament regime story that no era split here separates cleanly, and "
        "2013 and 2018 contribute +3.49% and +4.29% per-year means at h=5 against a +1.202% "
        "average."),
    "script": "scratch/pitch_checks/2026-09-07/03_pricestate_s4_sector_washout.py",
    "source": "near_miss",
    "expires": "2027-03-07",
    "note": "2026-09-07: parked the day it was found. Live at r21 2.8 and not traded, pending the reference class.",
})

entries.append({
    "added": "2026-09-07",
    "title": "Long HYG at the first close back from an extended market closure",
    "cell": "market-holiday closure x credit - the class the closure anchor had never been run on",
    "trigger": (
        "THE STATE FLIPPING, and it is the opposite of the state that generated it. The bare "
        "cell is real: 130 anchors after a >= 4 calendar-day NYSE closure, MOC h=5, +0.275%, "
        "66.2% hit, t 3.26 against an ordinary 3-day-weekend control of +0.094%, era-stable "
        "(pre-2013 +0.352 / 2018+ +0.229 / 2021+ +0.241), surviving a month-turn/coupon attack, "
        "top-5 moves only 23% of total, and only 0.38 correlated with the LQD post-closure cell "
        "and about 0.00 with IEF and TLT. Three independent constructions found it the same "
        "morning: the 697-cell closure survey, a Labor-Day-only seasonal grid at 73.7% on 14 of "
        "19, and a nearest-neighbour analogue lane that used no calendar information at all and "
        "made HYG the only instrument positive in all six of its constructions. WHAT KILLED IT "
        "is that the closure gate filters nothing in the state that was live: with HYG within "
        "1% of its 252d high AND in the calm realised-vol tercile the cell pays +0.064% (n=40) "
        "against +0.075% for an ordinary weekend in the same state and +0.069% for all days in "
        "it, a closure excess of -0.012% at Welch t -0.12, while the whole +0.267% excess "
        "(t 1.93) lives in the other half. TURNS ON when HYG is more than 1% BELOW its 252-day "
        "high OR its 21-day realised vol is above 4.4% annualised. Two debts it still owes even "
        "then. (1) A CREDIT-SPECIFIC RESIDUAL, which this family has now failed to produce five "
        "separate times: HYG = -0.014% + 0.189*IEF + 0.446*SPY, so +0.196% of the +0.275% is "
        "beta and the residual is 68-62 at sign p 0.331 (20-20 in the near-high bucket), while "
        "raw SPY pays +0.370% on the identical 130 anchors - a lower-return, higher-cost, "
        "0.43-beta way to buy the post-holiday equity drift. (2) A MECHANISM, because accrual "
        "make-up is falsified inside its own window: the first session back is -0.122% at "
        "t -2.62 against an unconditional +0.021%, the worst session in the neighbourhood, and "
        "45% of the 5-day total arrives on hold day 2. That is reversal, not carry. One benign "
        "finding worth keeping: corr(HYG 5d, SPY 5d) is +0.598 on the anchors, so 25% NAV of "
        "HYG at beta 0.43 is +10.8% NAV SPY-equivalent and would very nearly neutralise the "
        "Event Sleeve's T2 short-SPY 10% leg."),
    "script": "scratch/pitch_checks/2026-09-07/08_hyg_closure.py",
    "source": "near_miss",
    "expires": "2027-09-07",
    "note": "2026-09-07: parked the day it was found. HYG -0.41% off its high at 2.6% realised vol = the dead bucket.",
})

save_watchlist(wl)
print(f"watchlist: {len(entries)} active entries, 3 added, 4 notes updated")
