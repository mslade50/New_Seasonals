"""Post-publish watchlist maintenance for 2026-08-17.

Adds today's two near-misses, rewrites the W6 note with the finding that raises
its bar, and leaves every other entry's trigger untouched (only the dated
verdict note moves). Nothing here is a trigger change: a verdict is an
observation, not a re-specification.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa

ASOF = "2026-08-17"

NEW = [
    {
        "added": ASOF,
        "title": "Long TLT on the NOVEMBER month-position effect",
        "cell": "rates seasonal (month-of-year)",
        "trigger": (
            "the CALENDAR, and this one parks to a date rather than to a level. "
            "At matched trading-day-of-month 4-12, h=10 lag-1, November pays "
            "+1.590% over 24 years at a 20-4 year record (sign p 0.00077) and "
            "ranks 1 of 12 months. It is the live sibling of the August cell "
            "that died on 2026-08-17: August is 2018-2025 -0.013% at 4 of 8 "
            "years, November is 2018-2025 +2.093% at 8 for 8 (sign p 0.0039) "
            "and 2021-2025 +2.325% at 5 for 5. Charged in full for the "
            "12-month scan it came out of, Bonferroni gives 0.0093. The "
            "duration-neutral residual against IEF at the measured 1.914 beta "
            "is +0.462% full-sample and +0.358% for 2018+, so it is not purely "
            "a levered-IEF trade. TURNS ON in the entry window of trading days "
            "4 to 12 of November 2026, roughly 2026-11-05 to 2026-11-17. "
            "Before it trades, two debts: check the rate regime first, since "
            "the August version died precisely on the rising-yield side "
            "(+0.496% rising against +1.409% falling) and a bond-bull proxy "
            "can migrate month and still be a bond-bull proxy; and re-run the "
            "month-of-year table on data through October so the 8-for-8 is "
            "not being read at its own peak."
        ),
        "script": "scratch/pitch_checks/2026-08-17/r1_verify_august_and_november.py",
        "source": "stand_down",
        "expires": "2026-12-01",
    },
    {
        "added": ASOF,
        "title": "Short SPY with the index at a 52w high while the long end sits at a 52w low",
        "cell": "rates x us_large price-state",
        "trigger": (
            "COST on the de-concentrated form. The joint state (SPY within "
            "0.5% of its 52w high AND TLT within 1% of its 52w low) is live "
            "today at -0.20% and 0.15%. Its h=5 short side is 9-2 with sign p "
            "0.0327, the only pulse in the cross-asset conditioner lane on "
            "2026-08-17. TURNS ON when the de-concentrated form clears 5x "
            "cost: dropping the top 2 episodes (2018-10-03, 2021-02-24, "
            "together 96% of the +7.75pp total) leaves +0.039% per episode "
            "over the remaining 8, which is 1.3x a 3 bp round trip. Two "
            "standing caveats that survive any trigger: the sign INVERTS one "
            "horizon earlier at h=3 (-0.242% on 3-7, sign p 0.945), so h=5 is "
            "a knife edge rather than a plateau; and the threshold grid decays "
            "monotonically as the TLT rung loosens (+1.157 / +0.775 / +0.345 / "
            "+0.031), which says the level of the gate is doing the work "
            "rather than the joint state. The two legs are near-independent "
            "(corr 0.176). Re-measure after each new instance of the joint "
            "state; two more episodes of the observed size move the "
            "de-concentrated mean directly."
        ),
        "script": "scratch/pitch_checks/2026-08-17/a4c_c12_gateb_short_side.py",
        "source": "stand_down",
        "expires": "2027-02-17",
    },
]

# Dated verdicts for entries examined today. Trigger text is NEVER rewritten.
NOTES = {
    "Long TLT from the NFP close": (
        "2026-08-17 verdict: PASS, unchanged and still structurally "
        "unreachable. First non-midterm NFP is 2027-01; the next print "
        "(2026-09-04) is midterm and 14 td out, beyond the 10 td maximum "
        "horizon. NEW DEBT from today: this entry's parent lane is now on "
        "record as month-position rather than event (the bare August TLT "
        "seasonal is 2018-2025 -0.013%), so when the 2027-01 NFP arrives the "
        "cell owes a month-of-year control before it owes anything else."
    ),
    "Credit-quality divergence": (
        "2026-08-17 verdict: PASS, unchanged. State still live (HYG 0.10% off "
        "its 52w high, LQD 0.75% off its 52w low) and still the cluster begun "
        "2026-07-22, so the count is still 4 declustered episodes with three "
        "in 2018 and today would again be a mid-cluster entry."
    ),
    "Long SVXY overnight into the CPI print": (
        "2026-08-17 verdict: PASS, still deferred with cause. Next CPI is "
        "2026-09-11, 18 td out and unreachable; the re-measure is owed at the "
        "2026-09-10 run. Adjacent evidence from today that sharpens it: the "
        "term-structure state is confirmed a LAGGING marker (placebo offset "
        "-10 pays +5.433% at an 85% hit against the true anchor's +1.672%), "
        "so when the overnight cell is re-measured, check whether the CPI "
        "anchor survives its own offset ladder and not only its cost bar."
    ),
    "Long GLD on a miner-led thrust": (
        "2026-08-17 verdict: PASS, divergence still absent (GDX 5d rank 40.5 "
        "against the >= 95 needed, GLD 46.4). Today's separate GDX work "
        "raises this entry's bar: the 15-name reference class gives a "
        "dispersion ratio of 0.97 and P(max >= GDX's excess) = 0.582, so any "
        "future miner-specific claim owes a reference class, not just a "
        "control."
    ),
    "Long XLE on a crude one-day thrust": (
        "2026-08-17 verdict: PASS, no one-day pop (USO 1d +1.26%, +0.31 ATR "
        "against the [5,6)% and >= 1.50 ATR the trigger needs). The 5-DAY "
        "complex thrust that WAS live today was checked separately and killed: "
        "the ATR-magnitude form is negative at every horizon 1-10. That "
        "supports rather than weakens this entry's magnitude-band framing."
    ),
    "Long TLT with the whole investment-grade complex": (
        "2026-08-17 verdict: PASS, and the BAR IS NOW HIGHER. The price rung "
        "switched back ON at Friday's close (TLT 0.15% off its 52w low, IEF "
        "0.95%, LQD 0.75%) but the freshness leg fails hard: today is depth 5 "
        "of an episode begun 2026-08-03 with the prior trigger 2 sessions ago, "
        "against the >= 10 required, and pooled depth>1 entries pay -0.629% at "
        "a 37.3% hit over N=59. Two NEW findings this entry must carry. (1) "
        "The parent it was meant to gate is dead: the bare August TLT seasonal "
        "is 2018-2025 -0.013% at 4 of 8 years and is a bond-bull fossil, so "
        "there is no longer a live seasonal for this rung to sharpen. (2) The "
        "rung SUBTRACTS where it has precedent at all: TLT alone within 1% of "
        "its 52w low crossed with the August window moves the parent -0.233pp "
        "at h=10 (6 days, 2 years) and -0.744pp at h=5, and the tight "
        "three-way rung leaves 2 qualifying days of 241."
    ),
    "Long SPY on a skew spike alone": (
        "2026-08-17 verdict: PASS, all three legs still fail. ^SKEW 5d rank "
        "86.1 against the >= 95 needed, SPY 0.20% off its 52w high against the "
        "> 1% needed, and the year is midterm against the non-midterm the "
        "trigger requires."
    ),
    "Fade a crude thrust out of a deep base": (
        "2026-08-17 verdict: PASS. USO 5d rank 82.9 against the >= 90 the "
        "trigger needs (63d rank 8.3 does clear its leg), and the post-2020 "
        "episode count is still 4 against the 8 required, which is structural "
        "rather than tape."
    ),
    "Long the medical-device thrust": (
        "2026-08-17 verdict: PASS. IHI 21d rank 97.6 against the 100 the "
        "trigger needs, and the reference-class gate (Cochran Q p 0.544 across "
        "27 sector ETFs) cannot move in two sessions. Today's GDX work is the "
        "third independent replication of this entry's own method."
    ),
    "Long China's five-day break": (
        "2026-08-17 verdict: PASS, and the FAMILY is now closed rather than "
        "just this instance. FXI's 5d rank 11.5 clears its leg and EEM's +1.48% "
        "clears, but the 21d rank is 60.7 against the >= 80 that defines the "
        "intact thrust. Separately, EWZ was tested a third time today and is "
        "wrong-signed at its own z10 depth (residual -0.806%), so "
        "'one market decouples from a risk-on thrust' should be treated as a "
        "dead family and this entry kept only for its residual-positive "
        "condition, which has never once been met."
    ),
    "An industry-wide five-day breadth washout": (
        "2026-08-17 verdict: PASS, untested-form leg still outstanding. Today "
        "produced an independent replication of this entry's core finding "
        "outside insurance: XRT's 5d washout alone pays +0.290% at h=5 and "
        "adding the intact-63d gate takes it to -0.443%, with the broken-trend "
        "half at +0.483%. The inverter is now confirmed on two unrelated "
        "universes, which strengthens the BROKEN-trend direction this entry "
        "parks on."
    ),
}


def main() -> None:
    wl = load_watchlist()
    entries = wl.get("entries", [])
    stamped = 0
    for e in entries:
        for key, note in NOTES.items():
            if e.get("title", "").startswith(key) or key in e.get("title", ""):
                e["note"] = note
                stamped += 1
                break

    titles = {e.get("title") for e in entries}
    added = [n for n in NEW if n["title"] not in titles]
    entries.extend(added)

    # Prune: expiry is a date string; drop anything past it.
    keep, dropped = [], []
    for e in entries:
        exp = str(e.get("expires", ""))
        (dropped if exp and exp < ASOF else keep).append(e)

    wl["entries"] = keep
    wl["expired"] = dropped
    save_watchlist(wl)
    print(f"stamped {stamped} verdict note(s); added {len(added)}; "
          f"pruned {len(dropped)}; {len(keep)} active")
    for e in keep:
        print(f"  - [{e['added']}] {e['title']}")


if __name__ == "__main__":
    main()
