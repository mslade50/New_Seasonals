import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

HERE = Path(__file__).resolve().parent
reg = ROOT / "data/pitch_negative_registry.md"
text = reg.read_text(encoding="utf-8")
block = (HERE / "_registry_append.md").read_text(encoding="utf-8")
if "## 2026-09-29:" not in text:
    reg.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")

w = load_watchlist()
entries = w["entries"]
titles = {e["title"] for e in entries}
new = [
    {
        "added": "2026-09-29",
        "title": "Long crude into payrolls after a 21-day crude thrust, month turn at or before entry",
        "cell": "nfp x energy, the 21d neighbour of the killed 63d-thrust run-in",
        "trigger": "A LAYOUT AND A RANK. USO 21d rank >= 75 on the k=-4 close, long USO from the k=-3 close to the NFP close, pays +1.574% on 43-23 (sign p 0.009) against +0.082% for the 21d thrust state without a print; 1st of 18 on the anchor ladder, monotone from rank 60 to 90, into CPI +0.04%; found on an ~8-cell gate walk, so charge it to sign p ~0.07. It dies when a month-end falls inside the hold (-0.397% on 8-6; the 2026-10-02 layout, month-end two sessions before the print, +0.043% on 6-3) and pays +2.105% on 35-17 when the turn falls at or before entry (Welch t 2.10). TURNS ON at a payrolls date whose k=-3 entry close falls after the month-end close, with USO 21d rank >= 75 on the k=-4 close; the next is the 2026-11-06 print (signal bar 11-02, entry the 11-03 close). A further neighbour for the 10-01 morning only: entry at the 10-01 close for one session to the 10-02 NFP close, gated on USO 21d rank >= 75 on the 09-30 close, pays +0.743% on 40-25 (p 0.041) and +0.752% on 5-3 in this layout; it is a neighbour of a neighbour, so pitch it only with that walk charged.",
        "script": "scratch/pitch_checks/2026-09-29/kB_c8d_uso_nfp_me.py",
        "source": "stand_down",
        "expires": "2026-11-10",
    },
    {
        "added": "2026-09-29",
        "title": "EWZ against EEM across a Brazilian first-round vote, poll-miss side only",
        "cell": "Brazil election x international",
        "trigger": "A POLL-MISS MEASURE THE REPO LACKS. From the k=-4 close to the first post-vote close the pair pays +7.41% on 5-0 (sign p 0.031, permutation P < 0.001) and the ^BVSP run-in is 6-0 (p 0.016), but uncertainty resolution is falsified (positive run-in, runoffs 3-3 with the pair at +0.004%, municipal years 1-4). The record tracks the market-favoured challenger beating first-round polls (2014, 2018, 2022; 2002 -6.0%). TURNS ON only if the repo gains final-week first-round polling so a side_fn can be written and tested on the six prior votes without re-using them to pick the sign; the 2026-10-25 runoff is not a firing (runoffs are the dead half). Next first round 2030.",
        "script": "scratch/pitch_checks/2026-09-29/kB_c4b_ewz_vote_r2.py",
        "source": "stand_down",
        "expires": "2026-10-20",
    },
]
for e in new:
    if e["title"] not in titles:
        entries.append(e)
for e in entries:
    if e["title"].startswith("Short silver after the whole metals complex breaks together"):
        note = ("2026-09-29 verdict: CHECK, FIRED. Out-of-sample first break 1 of 10 on the 09-28 bar "
                "(GLD -3.94%, SLV -5.49%, GDX -5.36%; no faithful break 09-21..09-25). Score short SLV from "
                "the 09-29 close to the 09-30 close on the 09-30 morning (SLV 54.95 on 09-28). Silver fell "
                "1.50pp LESS than its GLD beta that day (85th pctile of first breaks), the bucket that pays "
                "less (w_28_first_break.py, kA_c2b_slv_gld_pair.py).")
        if "2026-09-29" not in e.get("note", ""):
            e["note"] = (e.get("note", "") + " | " + note).strip(" |")
    if e["title"].startswith("Long SVXY hedged against SPY after a bond-vol spike"):
        note = ("2026-09-29 verdict: PASS by 0.6 bp. SPY -0.744% against the -0.75% leg; ^MOVE +6.06% is the "
                "92.2nd pctile on the script's full-history quantiles and 89.99th on 2018-03+ quantiles. The "
                "damage-half band cell before the walk charge clears +0.35% only at h=3 (+0.394%; h=1 +0.291%, "
                "h=2 +0.213%). (w_64_band_damage.py)")
        if "2026-09-29" not in e.get("note", ""):
            e["note"] = (e.get("note", "") + " | " + note).strip(" |")
w["expired"] = []
save_watchlist(w)
print(len(entries), "entries")
