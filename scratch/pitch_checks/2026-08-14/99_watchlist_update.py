"""Post-publish watchlist maintenance for 2026-08-14.

Appends today's one near-miss and stamps each existing entry with the verdict
stage B1 owed it (see 00_surface_map.md section 4). Nothing expired and nothing
fired, so there is no prune.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

ASOF = "2026-08-14"

# Verdict stamped onto entries carried forward, keyed by a unique title fragment.
NOTES = {
    "Long TLT from the NFP close": (
        "2026-08-14 verdict: PASS, and structurally unreachable today rather "
        "than merely not live. The trigger needs a non-midterm NFP (first is "
        "2027-01); the next NFP is 2026-09-04, both midterm and 15 td out, "
        "beyond the 10 td maximum pitch horizon."),
    "Credit-quality divergence": (
        "2026-08-14 verdict: PASS, unchanged. State still live (HYG 0.00% off "
        "its 52w high, LQD 1.16% off its 52w low) and still the cluster that "
        "began 2026-07-22, so the count is still 4 episodes with three in 2018 "
        "and today would be a mid-cluster entry."),
    "Long SVXY overnight into the CPI print": (
        "2026-08-14: the owed re-measure is still deferred with cause. Next CPI "
        "is 2026-09-11, 20 td out and unreachable from any legal horizon; owed "
        "at the 2026-09-10 run as the entry says. Adjacent evidence from today "
        "that this entry must carry: SVXY went -1x to -0.5x on 2018-02-28, so "
        "any sample pooled across that date is two instruments (pre-break daily "
        "sd 4.60% and worst day -88.41%, post-break 2.41% and -21.43%). Confirm "
        "the 100-event overnight sample is post-break before re-measuring."),
    "Long GLD on a miner-led thrust": (
        "2026-08-14 verdict: PASS, trigger not live and the divergence is "
        "absent in both directions. GDX 5d rank 70.2 against the >= 95 needed, "
        "GLD 68.7 against the < 95 needed."),
    "Long XLE on a crude one-day thrust": (
        "2026-08-14 verdict: PASS, no pop to fade or follow (USO 1d -1.78%). "
        "Separately reinforced today from the relative-value side: the standing "
        "XLE-over-USO 63d divergence is a bear-tape selector (trigger days "
        "below SPY's 200d 41.7% against a 20.3% base rate) and USO's roll decay "
        "is not shortable at these horizons (unconditional h=5 mean -0.003% "
        "with a POSITIVE median +0.251%)."),
    "Long TLT with the whole investment-grade complex": (
        "2026-08-14 verdict: PASS, and the PRICE RUNG HAS NOW SWITCHED OFF. TLT "
        "sits 0.82% off its 52w low against the <= 0.5% the tight rung needs, "
        "IEF 1.23% (needs <= 1.0) and LQD 1.16% (needs <= 1.0); Thursday's "
        "+0.58% TLT session ended the state. The freshness leg was failing "
        "anyway at 4 trigger days since 2026-08-03. The month-of-year debt from "
        "2026-08-13 still stands before this trades."),
    "Long SPY on a skew spike alone": (
        "2026-08-14 verdict: PASS, trigger not live on any of its three legs "
        "(^SKEW 5d rank 47.6 against >= 95; SPY 0.00% off its 52w high against "
        "the > 1% below required; 2026 is midterm). IMPORTANT ADDITION from "
        "today, which examined the OPPOSITE pole: a LEVEL percentile computed "
        "on a trailing 252d window is still a rank trap on this series, because "
        "^SKEW's median has drifted 114.04 (2000-04) to 143.11 (2026). When "
        "this entry fires, quote the spike's level against full history and "
        "2018+ as well, not against the trailing year alone."),
    "Fade a crude thrust out of a deep base": (
        "2026-08-14 verdict: PASS, still 4 post-2020 episodes against the 8 "
        "required, and there is no thrust to fade (USO 1d -1.78%)."),
    "Long the medical-device thrust": (
        "2026-08-14 verdict: PASS. The trigger is a reference-class condition "
        "(Cochran Q p < 0.05 across 27 sector ETFs, measured 0.544) plus "
        "episode-first freshness, and a structural gate cannot flip in one "
        "session. Today's run reinforced the method: an industry-level version "
        "of the same shape gave Q 6.49 on 9 df at p 0.690 with the named "
        "industry ranking 2 of 10."),
    "Long China's five-day break": (
        "2026-08-14 verdict: PASS, two of three legs fail. FXI 5d rank 15.5 "
        "clears the <= 20 leg, but the 21d rank is 61.9 against the >= 80 that "
        "defines the intact thrust, and EEM's +2.55% 5d return clears. The "
        "residual condition was never reached."),
}

NEW = {
    "added": ASOF,
    "title": ("An industry-wide five-day breadth washout, taken with the trend "
              "BROKEN instead of intact"),
    "cell": "sectors / industry breadth price-state",
    "trigger": (
        "which half of the gate is live, plus an unproven tradeable form. The "
        "breadth washout ALONE pays +0.885% at h=5 over 225 episodes (t 2.405) "
        "against a +0.280% all-days baseline, and the parent cell is +0.917% "
        "over 233 episodes (t 2.589). The whole of that lives in the NOT-INTACT "
        "half: adding the intact-63d-uptrend gate takes the cell to -0.789% on "
        "5-8, which is where 2026-08-14 sat (insurer median 63d rank 82.9). "
        "TURNS ON when >= 70% of a coherent industry universe sits at a 5-day "
        "rank <= 20 while the median 63d rank is BELOW 70, AND a <= 4-name "
        "selection rule clears the alphabetical placebo on the LONG side, which "
        "has not been tested. That second leg is not optional: on the short "
        "side the four alphabetically-first names returned +1.568% against "
        "+0.905% for the four most washed, so a selection rule that ignores the "
        "signal beat the one that uses it, and the pitch grammar caps an idea "
        "at four legs. Standing caveat that survives any trigger: the reference "
        "class across 10 industry groups gives Cochran Q 6.49 on 9 df (p 0.690, "
        "I-squared 0.0%) with a common excess of +0.819pp, so this is an "
        "any-industry effect and no industry label carries information. Note "
        "also that the intact-gated cell is NOT a bad-tape artifact in reverse: "
        "its trigger days sit below SPY's 200d only 4.5% of the time against a "
        "25.4% base rate, so the gate selects good tape and the cell still "
        "loses in it."),
    "script": "scratch/pitch_checks/2026-08-14/b2b_insurance_round2.py",
    "source": "stand_down",
    "expires": "2027-02-14",
}


def main() -> None:
    wl = load_watchlist()
    entries = wl.get("entries", [])

    stamped = 0
    for e in entries:
        for frag, note in NOTES.items():
            if frag.lower() in str(e.get("title", "")).lower():
                e["note"] = note
                stamped += 1
                break

    titles = {str(e.get("title", "")).strip().lower() for e in entries}
    if NEW["title"].strip().lower() not in titles:
        entries.append(NEW)
        print(f"appended: {NEW['title']}")
    else:
        print("near-miss already present, not duplicated")

    wl["entries"] = entries
    save_watchlist(wl)
    print(f"stamped {stamped} carried-forward entries with a "
          f"{ASOF} verdict; {len(entries)} active, 0 pruned "
          f"(none expired, none fired)")


if __name__ == "__main__":
    main()
