"""Post-publish watchlist maintenance for 2026-09-09.

Appends today's three near-misses, each carrying the number that turns it on.
Nothing expired today and nothing fired, so no prune. Run once, after publish.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

ADDED = "2026-09-09"

NEW = [
    {
        "added": ADDED,
        "title": "Long TLT from the PPI release close, with the ten-year at a 252-day high",
        "cell": "ppi x rates, the POST-release anchor",
        "trigger": (
            "A CORRELATED-FAMILY PERMUTATION, and note the anchor is the RELEASE "
            "close rather than the pre-print close this tab can place. The cell: long "
            "TLT entered MOC at the close of the PPI release session with ^TNX at a "
            "252-day high pays +1.352% at h=3 over 9 declustered episodes, 8-1, sign p "
            "0.0195, era-stable at pre-2018 +1.490% (4 of 4) and 2018+ +1.242%, midterm "
            "+0.996% against non-midterm +1.637%, and 45.1x a 3 bp round trip. WHAT "
            "BLOCKS IT is that it is the top rung of an 11-offset ladder x 2 vehicles x "
            "3 horizons: the charged permutation is 0.2682 against an uncharged 0.0055, "
            "and that null treats 66 correlated cells as independent, so it over-charges "
            "a ladder whose rungs share most of their observations. TURNS ON when a "
            "permutation that preserves the ladder's correlation structure puts the "
            "observed +1.352% under 0.05, or at +4 new declustered episodes with the "
            "record still at or above 80%. WHY IT COULD NOT SHIP on 2026-09-09 even at "
            "conviction: the anchor for this PPI is the 2026-09-10 close, and the pitch "
            "tab places on the morning it is published. THE PRE-RELEASE FORM IS DEAD and "
            "must not be substituted: the true k=-2 anchor ranks 8 of 11 at h=1 and 4 of "
            "11 at h=3 on the same ladder, and gap-share falsifies it outright, with the "
            "release session itself paying -0.115% on IEF (-28% of the h=3 hold) and "
            "-0.306% on TLT (-55% of a +0.556% hold). Duration SELLS OFF on the print "
            "and makes it back afterwards, which is why the informative rung is k=+1. "
            "One differentiation already settled, do not re-derive it: this is NOT a "
            "re-skin of watchlist 41, Jaccard against the DBC-252d-high mask is 0.0358 "
            "with P(COMMOD | RATE) = 0.092, and the surviving direction is the opposite "
            "one. One definition caveat: the headline moves 0.188pp between "
            "decluster-then-filter (+0.406%, n=10) and filter-then-decluster (+0.218%, "
            "n=9) on the pre-release form, so fix the order before quoting anything."
        ),
        "script": "scratch/pitch_checks/2026-09-09/c12c_postprint_nearmiss.py",
        "source": "stand_down",
        "expires": "2027-09-09",
        "note": "",
    },
    {
        "added": ADDED,
        "title": "Long SPY with high yield at a 252-day high on the session the ten-year prints one too",
        "cell": "credit x rates -> us_large, the equity leg of a cell with no credit residual",
        "trigger": (
            "THE INDEX'S OWN DEPTH, and the live tape sat outside the sample that "
            "generated it. The cell (HYG within 0.5% of its trailing-252 high AND ^TNX "
            "at a 252-day high) pays SPY +0.558% at h=3 over 7 episodes, 6-1, sign p "
            "0.0625, bootstrap P(mean<=0) 0.018, with the credit gate worth +0.727pp "
            "over the rates parent whose complement pays -0.364%, and a MONOTONE dose "
            "response across the credit rung (0.25% +0.601 / 0.5% +0.558 / 1.0% +0.372 "
            "/ 2.0% -0.081%). WHAT KILLED IT is depth: 6 of the 7 episodes had SPY at or "
            "within 0.22% of its OWN 252-day high, five of them exactly at it, while the "
            "1.0-2.0%-off band holds exactly ONE episode and that episode is one of the "
            "two carrying 66% of the total. On 2026-09-09 SPY was 1.53% off. TURNS ON at "
            "a cell day with SPY ALSO within 0.5% of its own 252-day high, where the 6 "
            "matching episodes paid +0.467% at h=3 against a complement of +0.014%; and "
            "independently the vehicle x horizon permutation tested against the defended "
            "+0.558% must fall under 0.05, it is 0.7688 (uncharged 0.2712). TWO CAVEATS "
            "NO TRIGGER CURES. (1) The horizon is a spike, not a plateau: h=2 +0.599, "
            "h=3 +0.558, h=4 -0.092, h=5 -0.713% with a worst episode of -5.967%, so "
            "only h=3 is the object. (2) There is NO credit-specific residual and this "
            "is the SIXTH consecutive failure of the family to produce one: HYG = "
            "-0.001% + 0.189*IEF + 0.395*SPY (R-squared 0.477), the residual vehicle "
            "pays -0.107% at h=1 on 4 of 15 (t -2.465) and -0.180% at h=3 on 2 of 8, and "
            "the HYG gate against the rates parent is worth -0.001pp with the discarded "
            "complement paying MORE. Pitch the equity leg or nothing. ONE THING THAT "
            "PASSED, unlike watchlist 24: the fragility dial is INSIDE this cell's "
            "sample, which reaches ma10(63d) 87.9 with 4 of 12 days at or above 70, "
            "where watchlist 24's maximum ever observed was 68.0."
        ),
        "script": "scratch/pitch_checks/2026-09-09/c11b_credit_rates_r2.py",
        "source": "stand_down",
        "expires": "2027-09-09",
        "note": "",
    },
    {
        "added": ADDED,
        "title": "Long SVXY on the settle session when a VIX expiry lands on an FOMC decision date",
        "cell": "vix_expiry x fomc_decision collision x volatility, the settle-session rung",
        "trigger": (
            "NEW POST-2018 COLLISIONS, because the era break IS the instrument. The cell "
            "pays +1.574% over 25-9 at sign p 0.004 and it is the rare volatility cell "
            "that CLEARS the mandatory SPY-residual rule outright: alpha +1.628% at t "
            "5.91 against SPY (R-squared 0.797, the beta term explaining only -0.055%), "
            "so this is genuinely volatility-specific rather than levered equity wearing "
            "a vol label. WHAT KILLED IT is that pre-2018 pays +3.943% on 13-2 at t 5.06 "
            "while 2018+ pays -0.297% on 12-7 at t -0.48, and February 2018 is exactly "
            "when SVXY re-levered from -1.0x to -0.5x, so the half carrying the number is "
            "a product that no longer trades. The two most recent collisions are both "
            "losers (2026-03 -3.98%, 2026-06 -2.17%). TURNS ON when the 2018+ subsample "
            "alone reaches a positive mean with a record at or above 60% on at least 16 "
            "collisions, i.e. +4 new ones beyond today's 12, measured on the -0.5x "
            "vehicle only and never pooled with the pre-2018 half. Collisions run about "
            "20.2% of FOMCs (45 of them in 2000-2027), so this accrues roughly once a "
            "year. THREE THINGS ALREADY SETTLED, do not re-run them. (1) The placebo "
            "ladder ranks the true anchor 1 OF 11 on both the run-in and the settle "
            "session, so the anchor is real. (2) The RUN-IN rung is dead and is not this "
            "entry: it splits non-midterm +1.504% (t 2.87) against midterm -1.753% on "
            "4-6, reduces to the pre-FOMC drift's midterm inversion (all-FOMC midterm "
            "gap -1.084pp against the collision's -3.257pp, same sign amplified), and its "
            "only live direction is the Event Sleeve's own T2 short. (3) The settlement "
            "FLOW mechanism cannot be falsified in this repo at all: there is no dealer "
            "gamma, no open-interest history and no roll positioning, and the option_* "
            "files start 2026-08-05. Any revival owes a mechanism that local data can "
            "test, or it ships graded as unverified on mechanism."
        ),
        "script": "scratch/pitch_checks/2026-09-09/a3c_c8_svxy_settle.py",
        "source": "stand_down",
        "expires": "2027-09-09",
        "note": "",
    },
]


def main() -> None:
    wl = load_watchlist()
    entries = wl.get("entries", [])
    titles = {e.get("title") for e in entries}
    added = 0
    for e in NEW:
        if e["title"] in titles:
            print(f"SKIP already present: {e['title']}")
            continue
        entries.append(e)
        added += 1
    wl["entries"] = entries
    wl["asof"] = ADDED
    save_watchlist(wl)
    print(f"appended {added} near-miss entries; watchlist now holds "
          f"{len(entries)} active, {len(wl.get('expired', []))} expired")


if __name__ == "__main__":
    main()
