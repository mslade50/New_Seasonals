"""Post-publish watchlist maintenance for 2026-08-31.

Appends today's three `closest` near-misses with the number each turned on,
and prunes entries whose window has passed or whose cell was adjudicated in
this morning's sweep.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

wl = load_watchlist()
entries = wl.get("entries", [])
print("before:", len(entries))

NEW = [
    {
        "title": "Short silver after the whole metals complex breaks together",
        "cell": "metals, complex-break continuation on the highest-beta member",
        "trigger": (
            "THE LAG PROFILE, i.e. a mechanism rather than a level, and it is the "
            "hardest arm on this list. The cell passes everything the usual battery "
            "asks: gate attribution is decisive (the three-name conjunction pays "
            "+0.531% at h=1 against +0.031% for SLV breaking alone, the discarded "
            "half -0.091% and the anti-cell -0.278%), the local +/-126td control "
            "clears at welch t +2.21, the record is 67-52 at sign p 0.019 scored "
            "against silver's OWN 46.35% down-rate, declustering is stable at gap "
            "5/10/21, and the GLD-beta residual survives at +0.322pp, so it is NOT "
            "the closed 'second metals leg is size' objection. It dies because the "
            "effect has no shape: lag=0 pays +0.039%, lag=1 +0.516%, lag=2 +0.035%. "
            "One session wide, starting a session LATE, which no forced-deleveraging "
            "continuation predicts, and the naive no-lag look being 13x SMALLER than "
            "the tradeable one is backwards from every other cell in this repo. The "
            "entry-day direction is also irrelevant (corr -0.033) and runs against "
            "the story, since the cell pays +0.867% on 28-15 when silver BOUNCES "
            "over 1% on the entry day against +0.573% on 25-25 when it keeps "
            "falling. TURNS ON at either of two numbers: (a) the live-depth bucket "
            "(an SLV break of -4% or worse, where 2026-08-28's -4.38% sat) reaching "
            "46-36 from its current 35-36, which is sign p 0.049 against the 46.35% "
            "down-rate; or (b) the six-family common excess reaching +0.229% from "
            "+0.132%, or the raw metals excess +0.936% from +0.566%, either of which "
            "lifts the empirical-Bayes shrunk estimate (weight 0.21, tau-squared "
            "0.0198) past 5x a 6 bp round trip from today's 3.7x. NEITHER FIXES THE "
            "LAG PROFILE, so a future instance must ALSO show the effect at lag=0 or "
            "lag=2 before it is tradeable on a stated mechanism. Two standing "
            "caveats: 2026 alone is 32% of the 20-year h=1 total (+19.71pp of "
            "+61.38pp) and ex-2008/2020/2026 the cell is +0.204% on 45-39; and "
            "Friday triggers are the only negative weekday (-0.340%, 10-13) while "
            "month-end entry sessions pay -0.474% on 4-4, both of which described "
            "2026-08-31 exactly. Development work is already done and reusable: MOC "
            "is the only entry that works (+0.596% per signal against +0.068% at a "
            "0.15 ATR close-anchored limit on a 76.9% fill), any target destroys it "
            "(1.0 ATR takes +0.596% to +0.381%), h=1 time-only, and losers averaged "
            "-1.85% by day 1 with no day 2 to recover in. See b4c2_c19_slv_dev.py."
        ),
        "script": "scratch/pitch_checks/2026-08-31/b4c_c19_slv_short_teardown.py",
        "source": "stand_down",
        "expires": "2027-08-31",
    },
    {
        "title": "Long duration with the ten-year at a yield high and bond vol MID-RANGE, not compressed",
        "cell": "rates, yield-level x bond-vol band",
        "trigger": (
            "AN EPISODE COUNT, and note the entry is the OPPOSITE of the candidate "
            "that produced it. The pitched cell was 'yield high AND bond vol "
            "COMPRESSED' and that conjunction beats neither parent (episode-vs-"
            "control -0.248% at welch t -0.72, midterm wrong-signed at -0.453% on a "
            "42.9% hit). What survived round 1 is the MID-RANGE band: with ^MOVE's "
            "trailing-252 LEVEL percentile in [40,50), TLT pays +1.064% at h=5 over "
            "7 episodes at a 100% hit, 35x cost, era-stable, and with a genuine "
            "PLATEAU across the yield gate (0.25 / 0.5 / 1.0 / 2.0 / 3.0% all pay "
            "+0.78 to +1.06%) rather than the knife edge that usually marks a mined "
            "threshold. It is blocked by multiplicity and by its own dose response. "
            "Charged for the grid actually walked -- 6 MOVE bands x 4 horizons x 2 "
            "vehicles, with the sign scan absorbed by taking |gate| -- the "
            "permutation gives P = 0.857 at h=5 and 0.305 at h=10, so the +1.269pp "
            "live-band gate is a BELOW-median draw from the distribution of the best "
            "cell under no effect. And the ladder is an inverted U peaking at pctile "
            "55-57 (Spearman -0.037), with the MOST compressed band [0,20) the WORST "
            "long-duration bucket at -0.809%, the opposite sign from any "
            "orderly-repricing story exactly where it should be strongest. Band "
            "neighbours are soft: [35,55) +0.073%, [40,55) +0.227%, [35,50) +0.358%. "
            "TURNS ON when the full-grid permutation P falls under 0.05, which at "
            "the observed effect size needs roughly 25-30 episodes in the live band "
            "against today's 7, i.e. about one qualifying episode every three years "
            "-- park it and expect nothing this decade. Two notes for whoever "
            "re-opens it: the cell is separately OUT OF SAMPLE on the fragility dial "
            "(its trigger-day maximum is 66.8 against 87.6 on 2026-08-31), and "
            "vehicle choice is free, since excess per unit of sd runs 0.520 TLT / "
            "0.633 IEF / 0.529 LQD."
        ),
        "script": "scratch/pitch_checks/2026-08-31/b2c_c2_fullgrid_and_dose.py",
        "source": "stand_down",
        "expires": "2028-08-31",
    },
    {
        "title": "The small-cap month-end OVERNIGHT in December, as a scan-charged seasonal",
        "cell": "us_small, month-position x month-of-year, overnight return",
        "trigger": (
            "A DATE, and a scan charge that has never been paid. This is the residue "
            "of a cell whose MECHANISM was falsified, so the arm condition is "
            "deliberately narrow. The parent object is new to this repo -- every "
            "prior month-end closure measured close-to-close, and nobody had "
            "measured Open[ME+1]/Close[ME-0] -- and its headline is real: IWM "
            "+16.80 bp of excess over its unconditional overnight, 64.6% hit against "
            "a 55.1% base rate, sign p 0.0004. The auction-reversal story is dead "
            "all the same: the reversal regression is WRONG-SIGNED on the one "
            "session that has the auction (IWM +0.081 on ME-0 against -0.079 on all "
            "sessions; SPY +0.194 against -0.131), and the 15-vehicle reference "
            "class is led by EEM (+21.2 bp) and EFA (+15.0 bp), two markets that are "
            "SHUT while the US closing auction prints and reopen overnight in Asia "
            "and Europe, with the family homogeneous (Cochran Q p 0.6875, I-squared "
            "0.0%) at a common excess of +8.26 bp. It is one market-wide overnight "
            "drift wearing fifteen labels. What is left is a calendar fact: DECEMBER "
            "pays IWM +53.13 bp of excess over 26 years at an 80.8% hit, 10.6x a "
            "5 bp round trip, against a full-year +16.80 bp and an August -6.05 bp. "
            "TURNS ON at a December ME-0 in a NON-MIDTERM year, i.e. 2027-12-31 at "
            "the earliest, AND ONLY IF the max-of-12 month permutation is run first "
            "and clears -- it currently reads P 0.476 for IWM, which is why this is "
            "parked rather than shipped. Do NOT re-pitch it as auction flow; that "
            "mechanism is closed. Two data facts worth keeping: ^GSPC is NOT a "
            "usable overnight instrument before ~2013 (Yahoo's synthetic open gives "
            "a median overnight of exactly 0.000 at a 25.0% up-rate), and the raw "
            "unadjusted overnight excess is LARGER than the adjusted one, so "
            "dividend contamination explains none of it."
        ),
        "script": "scratch/pitch_checks/2026-08-31/b1d_round2_refclass.py",
        "source": "stand_down",
        "expires": "2028-01-31",
    },
]

titles = {e.get("title") for e in entries}
for n in NEW:
    if n["title"] in titles:
        print("  already present, skipped:", n["title"])
        continue
    n["added"] = "2026-08-31"
    entries.append(n)
    print("  appended:", n["title"])

# Prune: windows that have passed, or cells adjudicated in today's sweep.
PRUNE = {
    # The ME-9 entry window closed 2026-08-18; 2026-08-31 was its EXIT date.
    "Long TLT into the month-end close, ungated, entered nine sessions before it",
    # Jackson Hole 2026 is behind us and the anchor cannot recur until Aug 2027.
    # The lane is closed on eight classes pre-speech and ten post-speech.
    "Long crude through Jackson Hole, entered six sessions before the conference",
    "Long high yield across the Jackson Hole speech, entered five sessions before it",
    # The ME-3 -> ME-2 session was 2026-08-26 -> 08-27 and passed unarmed;
    # superseded by today's overnight work on the same anchor.
    "The single small-cap session from three sessions before month-end to two",
    # Adjudicated and CLOSED today: definition fragility (the two percentile
    # conventions disagree at the identical rung, +0.934% vs +0.005%) plus a
    # 12-pair reference class with a NEGATIVE common excess of -0.121%.
    "Long OIH outright, no short leg, at a 63-day services-versus-E&P extreme",
}
kept, dropped = [], []
for e in entries:
    (dropped if e.get("title") in PRUNE else kept).append(e)
for e in dropped:
    print("  pruned:", e.get("title"))

wl["entries"] = kept
wl["expired"] = wl.get("expired", [])
wl["expired"].extend(
    {"title": e.get("title"), "cell": e.get("cell"), "retired": "2026-08-31",
     "why": "window passed or cell adjudicated in the 2026-08-31 sweep"}
    for e in dropped
)
save_watchlist(wl)
print("after:", len(kept), "| expired total:", len(wl["expired"]))
