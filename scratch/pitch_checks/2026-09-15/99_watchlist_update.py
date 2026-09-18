"""Post-publish watchlist update, 2026-09-15 stand-down.

W25's still-falling leg fired and its family form was killed on firing, so its
trigger is rewritten. W14, W20 and W22 get dated verdict notes (a state fired,
both numbered arms cleared, a 0.79-point miss). Two near-misses from the
stand-down's closest list are parked; the third closest (SMH) is W25 itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

TODAY = "2026-09-15"
EXPIRES_15TD = "2026-10-06"

d = load_watchlist()
entries = d["entries"]


def find(prefix: str) -> dict:
    hits = [e for e in entries if e["title"].startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def add_note(e: dict, text: str) -> None:
    e["note"] = ((e.get("note") or "") + " | " + text).strip(" |")


w25 = find("The leader's deep correction")
w25["trigger"] = (
    "KILLED 2026-09-15 ON FIRING, now OUT OF SAMPLE ONLY. The still-falling leg fired "
    "(SMH r63 1.19, r5 12.70 on 2026-09-14) and the family form died. (1) The parked 7-0 "
    "(+6.442%) reads r5 on each episode's FIRST day; the tradeable daily mask form on SMH is "
    "N=11, +3.26% at h=10, 8-3, sign p 0.113. (2) Out of sample on the 22 non-SMH members the "
    "cell pays +0.861pp over own drift on 86 episodes (46 date clusters, 27-19, p 0.151), but "
    "2021 carries 114% of that excess (a 252d >= +40% gate over-samples post-crash recovery "
    "years: 33 of 86 episodes), drop 2021 -0.193pp, drop 2021 and 2026 -1.001pp, pre-2018 "
    "-1.365pp. (3) The r5<15 conditioner RE-ANCHORS rather than filters ex-SMH: filtering "
    "-0.586pp, re-anchoring +1.166pp; the parent anchors it deletes pay +0.952pp and the ones "
    "it keeps pay -0.305pp at their own dates. TURNS ON only when BOTH hold on episodes "
    "signalled AFTER 2026-09-15: the ex-SMH, ex-2021 family excess at h=10 is >= +0.30pp over "
    "at least 20 new episodes, AND the kept-minus-deleted gap measured at parent anchor dates "
    "is >= +0.35pp. Standing caveats: dial max on trigger episodes 77.9 (non-SMH) and 74.7 "
    "(SMH) against 85.2 on 2026-09-14; SMH's live episode started 2026-09-09 at r5 79.4, the "
    "flat side of the split."
)
w25["script"] = "scratch/pitch_checks/2026-09-15/k1_c2_smh_family_b.py"
add_note(w25, "2026-09-15: still-falling leg FIRED and the family form was killed on firing "
              "(k1_c2_smh_family.py, _b, _c, _d); trigger rewritten to an out-of-sample arm.")

add_note(find("Long technology against healthcare after a rotation gap"),
         "2026-09-15 verdict: PASS. The subclass state FIRED on 2026-09-14 (XLV minus XLK "
         "+3.25pp, SPY -2.19% off its high, SPY Wilder ATR 0.82%), but it is a 2026-cluster "
         "episode and the arm needs three new winners OUTSIDE that cluster.")
add_note(find("Cross-sectional new-high breadth"),
         "2026-09-15 verdict: PASS. Both numbered arms cleared for the first time (SPY -2.19% "
         "off its high; raw-21d fragility 49.5), but the breadth leg underneath did not fire: "
         "zero of nine SPDRs within 0.25% of a 52w high (nearest XLE -1.19%).")
add_note(find("The utilities washout with the long end hit"),
         "2026-09-15 verdict: PASS, closest single-leg miss on the list. XLU r21 3.97 clears "
         "<= 5; TLT r21 25.79 against < 25, short by 0.79. Not re-derived at a looser rung.")

entries.append({
    "added": TODAY,
    "title": "Long SVXY into an FOMC decision after a pre-decision VIX re-bid, backwardated form only",
    "cell": "fomc_decision x volatility, re-bid at k=-2 split by term structure",
    "trigger": (
        "THE TERM STRUCTURE, then the charge for the split that found it. Long SVXY MOC at k=-1 "
        "to the decision close after ^VIX rose >= 5% on the k=-2 session pays +0.224% on 17-4 "
        "in the -0.5x era (sign p 0.004), beta-charged alpha +0.389% on 16-5 at beta 1.48, but "
        "the re-bid gate filters nothing (all-FOMC parent +0.237%, gate worth -0.018pp) and "
        "k=-2 ranks 6 of 11 on the placebo ladder. All of the alpha sits where VIX/VIX3M closed "
        ">= 0.90 at k=-2: +1.233% on 12-1 (sign p 0.002), alpha 11-2. Below 0.90 it is 5-3 at "
        "-1.42% and holds both decision-day blowups (2021-01-27 -10.54%, 2024-12-18 -8.66%). "
        "2026-09-14 read 0.887. TURNS ON at the next FOMC whose k=-2 session carries a >= 5% "
        "^VIX rise AND VIX/VIX3M >= 0.90 at that close, AND only if the split survives its own "
        "charge: the term threshold was found by a search, so walk 0.85 / 0.875 / 0.90 / 0.925 "
        "/ 0.95 and require the 0.90 cell's alpha sign p to hold at <= 0.05 after that charge. "
        "Standing notes: the midterm members are 5-0 but every one had a dose >= 9.4%, term "
        ">= 0.91 and dial <= 32 (live 85.2); pre-2018 short ^VIX in the state paid LESS than its "
        "complement (+1.97% against +2.71%), so the re-bid decays less, not more, on the "
        "non-levered read; the collision subset (FOMC on a VIX expiry) is +0.029% on 6-1."
    ),
    "script": "scratch/pitch_checks/2026-09-15/k3_c3_vix_rebid_b.py",
    "source": "stand_down",
    "expires": "2026-12-10",
    "note": "2026-09-15: closest #1 in the stand-down. Next anchors: FOMC 2026-10-28 "
            "(k=-2 session 2026-10-26) and 2026-12-09 (k=-2 2026-12-07).",
})

entries.append({
    "added": TODAY,
    "title": "Short a large bank against XLF after a non-earnings intraday slide of 1.5 ATR on a calm tape",
    "cell": "banks, single-name information shock continuation vs the sector ETF",
    "trigger": (
        "THE REFERENCE CLASS, then the modern era. Short the name against XLF at beta after a "
        "non-earnings, intraday-led drop >= 1.5 Wilder ATR (prior-day ATR) on a session SPY "
        "falls less than 1% pays +0.316% at h=5 over 108 episodes, 66-42, two-sided sign p "
        "0.026, +0.422pp over the same-date universe, positive in 18 of 26 years, top-two "
        "episodes -3% of total, 6.3x a ~5 bp round trip. It dies because banks are not "
        "special: the same rule across 12 sector groups (159 names against their sector SPDRs) "
        "has a common excess of -0.047pp (Cochran Q 7.66 on 11 df, I-squared 0%), banks rank 1 "
        "of 12 at P(max group >= banks) 0.324, and non-financial names REVERT (-0.214% to the "
        "short at h=5 over 867 episodes, t -1.91). 2018+ pays +0.034% (t 0.10, 0.7x cost) "
        "against +0.510% before. TURNS ON when BOTH move: banks' max-of-12 P falls to <= 0.10 "
        "(or a structural reason is shown why bank information shocks drift while other "
        "sectors' revert), AND the 2018+ bank cell reaches >= +0.25% at h=5 (5x cost). "
        "Neighbours on file: 2 ATR +0.850% (16-11), 2.5 ATR -0.018%; gap-led +0.740% (29-7) is "
        "stronger than the intraday-led form; dropping the SPY > -1% gate takes it to -0.019%. "
        "The long side of the same shock is dead (earnings-print content, -0.449% at h=5 on the "
        "non-earnings parent)."
    ),
    "script": "scratch/pitch_checks/2026-09-15/k2_c1x_family.py",
    "source": "stand_down",
    "expires": EXPIRES_15TD,
    "note": "2026-09-15: closest #2 in the stand-down; live instance BAC -2.93 ATR on 2026-09-14 "
            "(gap share 0.11, break at 13:00 ET), with GS, MS and BNY also in the cell.",
})

save_watchlist(d)
print(f"watchlist: {len(entries)} active entries after update")
