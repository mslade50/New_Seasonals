"""Post-publish watchlist maintenance for 2026-08-28.

Appends the morning's two near-misses with the number each turned on, and
prunes anything past its expiry. Ten of the eleven kills leave NOTHING here by
design: a joint state that pays less than the plain state underneath it has no
threshold that rescues it, so parking one guarantees it is re-found and
re-killed (registry, method traps 2026-08-28).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from pitch_lab import load_watchlist, save_watchlist

TODAY = "2026-08-28"

NEW = [
    {
        "added": TODAY,
        "title": "Long IEF for one session out of the Jackson Hole close, non-midterm years only",
        "cell": "jackson_hole x rates, the POST-speech anchor",
        "trigger": (
            "THE CYCLE YEAR, and this parks to a date rather than to a level. Long IEF MOC the "
            "session after the conference, held one session, is the FIRST event anchor in this "
            "repo whose placebo ladder isolates k=0: the true anchor pays +0.228% (t 3.99, n=24) "
            "while every neighbouring offset runs -0.09% to +0.11%. That makes the ladder's "
            "record 9-for-10, not 9-for-9. Non-midterm years pay +0.292% at t 4.54 on a 14-4 "
            "record over 18 anchors; midterm years pay +0.037% at t 0.41 and a 33.3% hit over 6 "
            "(2002 -0.04, 2006 +0.09, 2010 +0.45, 2014 -0.02, 2018 -0.22, 2022 -0.03), which is "
            "3.7 bps against a 3 bps round trip. TURNS ON at year %% 4 != 2, i.e. the anchor on "
            "2027-08-27; 2026 %% 4 == 2 so it is blocked this year. THREE debts to clear before "
            "it ever trades, none of which a new observation fixes on its own. (1) The whole "
            "210-cell class-by-horizon grid fails a family-wise permutation at P 0.065, so the "
            "cell is search-contaminated and needs re-deriving forward rather than from that "
            "grid. (2) It is ONE duration bet wearing four labels: IEF/TLT forward-return "
            "correlation is 0.911 and IEF/^TNX is -0.812, so the three confirming cells are not "
            "independent evidence. (3) The lag-1 entry lands ME-1..ME-6 in 20 of 24 anchors and "
            "the all-months ME-2 control alone pays IEF +0.108% (t 4.72, n=289), so a third to a "
            "half of the cell is month position available twelve times a year with no event. "
            "Standing caveat: era decay is monotone, +0.293% pre-2013 to +0.204% 2013-2019 to "
            "+0.137% at t 1.05 and a 50.0% hit from 2020, and LQD clears only 4.4x cost."
        ),
        "script": "scratch/pitch_checks/2026-08-28/c6_post_jackson_hole.py",
        "source": "stand_down",
        "expires": "2027-09-15",
    },
    {
        "added": TODAY,
        "title": "The laggard that is STILL FALLING, pooled across 29 index and industry ETFs",
        "cell": "cross-asset price-state, the pooled parent of watchlist 30",
        "trigger": (
            "A LIVE READING ON THE WRONG SIDE, and it is the same number the SMH pitch got "
            "backwards on 2026-08-26. Pooled over 29 index and industry ETFs, 21-day rank >= 90 "
            "AND 63-day rank <= 10 AND 5-day rank < 15 pays +1.437% at h=10 over 53 episodes at "
            "t 2.15 and a 67.9% hit, and inside that sub-cell the 63-day gate genuinely earns "
            "its place, adding +0.705pp over the 21d-plus-5d form alone (+0.732%, N=213). In the "
            "already-bouncing half, 5-day rank >= 25, the same gate is worth +0.139pp and the "
            "cell pays +0.619% at t 1.35. This is a pooled confirmation of watchlist 30 at more "
            "than 3x its episode count, so the two should be read together and retired together. "
            "TURNS ON when a name holding the joint state prints a 5-day rank below 15; EEM is "
            "the only holder today and reads 63.1. TWO debts. (1) The PARENT is dead on its own "
            "terms: bare 21-day rank >= 90 pays +0.476% over 3,521 observations at t 6.43 while "
            "the joint cell pays +0.370% over 189 at t 0.88, so the 63-day clause subtracts "
            "-0.106pp overall and discards 95% of the population -- the sub-cell has to justify "
            "itself as the exception to its own parent's sign rather than as the top of it. "
            "(2) The rank and level forms of the 63-day clause disagree at Jaccard 0.10, with "
            "the t-63 roll-off bar exceeding the day's own bar on 31.0% of trigger name-days, "
            "and the LEVEL form (63d return <= 0) pays +0.667% over 1,067 observations at t 4.07 "
            "-- better populated and better signed than the rank form the cell is written on, so "
            "the definition is unsettled. Family homogeneity is the standing blocker: Cochran Q "
            "19.21 on 28 df at p 0.8915, I-squared 0.0%, common excess -0.228%."
        ),
        "script": "scratch/pitch_checks/2026-08-28/b2b_c4_split_attribution.py",
        "source": "stand_down",
        "expires": "2027-02-28",
    },
]

wl = load_watchlist()
entries = list(wl.get("entries", []))
expired = list(wl.get("expired", []))

titles = {str(e.get("title", "")).strip().lower() for e in entries}
for n in NEW:
    if n["title"].strip().lower() in titles:
        print("SKIP duplicate: %s" % n["title"])
        continue
    entries.append(n)
    print("APPEND: %s" % n["title"])

today = pd.Timestamp(TODAY)
keep, dropped = [], []
for e in entries:
    exp = e.get("expires")
    if exp and pd.Timestamp(exp) < today:
        e = dict(e, expired_on=TODAY)
        dropped.append(e)
    else:
        keep.append(e)

for d in dropped:
    print("PRUNE (expired %s): %s" % (d.get("expires"), d.get("title")))

wl["entries"] = keep
wl["expired"] = expired + dropped
wl["generated"] = "%s (post-publish, stand-down)" % TODAY
save_watchlist(wl)
print("\nwatchlist: %d active, %d expired" % (len(keep), len(wl["expired"])))
