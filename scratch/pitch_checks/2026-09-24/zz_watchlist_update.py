"""Post-publish: 2026-09-24 near-misses, the W28 rewrite and the W18 tally note."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "data" / "pitch_watchlist.json"
w = json.loads(path.read_text(encoding="utf-8"))
entries = w["entries"]


def find(prefix: str) -> dict:
    hits = [e for e in entries if e["title"].startswith(prefix)]
    assert len(hits) == 1, f"{prefix!r}: {len(hits)} matches"
    return hits[0]


def add_note(entry: dict, note: str) -> None:
    old = (entry.get("note") or "").strip()
    entry["note"] = f"{old} | {note}" if old else note


silver = find("Short silver after the whole metals complex breaks together")
silver["trigger"] = (
    "REWRITTEN 2026-09-24: THE FIRST BREAK, AND AN OUT-OF-SAMPLE RECORD. The old arm "
    "(a -4% depth leg plus a lag profile) is retired: the depth gradient runs backwards "
    "(SLV <= -3.5% +0.454%, <= -4.0% +0.412%, <= -4.5% +0.269% at h=1 lag 1) and no single "
    "firing can show a lag profile, so it could never be met. What the re-run found: the "
    "faithful cell (GLD, SLV, GDX each <= -2%) still pays the SLV short +0.557% at h=1 from "
    "the next close on 68-52, but ALL of it is FIRST breaks (no faithful break in the prior "
    "5 sessions): +0.639% on 63-46, sign p 0.011, +0.446% pre-2018 and +1.084% after, while "
    "repeat breaks reverse at -0.953% on 16-34. The split was found on 2026-09-24, so it is "
    "post-hoc. TURNS ON when the first 10 first breaks from 2026-09-24 onward go 8-2 or "
    "better (sign p 0.033 against SLV's 46.35% down-rate) with a mean of +0.30% or more at "
    "h=1 lag 1 (10x a 3 bp round trip). Until then each live first break is scored, not "
    "pitched."
)
silver["script"] = "scratch/pitch_checks/2026-09-24/kC_s1_slv_w28_b.py"
add_note(silver, "2026-09-24 verdict: CHECK, not a firing. SLV -4.23% and GDX -4.36% cleared, "
                 "GLD -1.80% missed its -2% leg by 0.20pp; that SLV-and-GDX-only configuration "
                 "pays the short -0.066% on 84-123. Arm rewritten to the first-break form. "
                 "Out-of-sample first-break tally: 0 of 10.")

curve = find("The duration-neutral curve position")
add_note(curve, "2026-09-24 verdict: CHECK, tally moved. OOS episode 1 (signal 09-09, 09-10 to "
                "09-22) realized -65.0 bps on the h=8 curve against the +22.1 bps mean "
                "needed. OOS episode 2 signalled at the 09-23 close (^TNX 5.114 at its 252 max, "
                "252-session change +97.1 bp): entry 09-24 MOC, exit 10-06 close. Episodes 2 "
                "and 3 must now sum to >= +131.3 bps.")

new = [
    {
        "added": "2026-09-24",
        "title": "Long SVXY hedged against SPY after a bond-vol spike below the extreme tail",
        "cell": "volatility x rates, the inversion of the 2026-08-18 long-vol MOVE-spike kill",
        "trigger": (
            "THE DOSE, and it charges for the band walk. MOVE-spike days (top decile of daily "
            "^MOVE moves) pay the SVXY residual against 1.48x SPY +0.226% at h=1 on 155-100 "
            "(sign p 0.0003), 3.2x a 7 bp pair; with VIX also up the top decile pays +0.210% at "
            "h=5 on 80-57, and the 90th-95th percentile band +0.514%. It inverts in the tail: "
            "top 2% with VIX up -0.689% at h=5 on 17-21, top 1% -0.843%, and 2026-09-23's "
            "+21.5% was the 99.7th percentile. The no-damage half pays LESS than the damage "
            "half at every horizon (the fear-without-damage inverter reproduces). TURNS ON at "
            "a MOVE one-day rise between the 90th and 97th percentile of daily moves on a "
            "session SPY fell more than 0.75%, IF that damage-half band cell, re-derived with "
            "the band walk charged, clears 5x a 7 bp pair (+0.35%) at h=1..3 on 2018-03+ data."
        ),
        "script": "scratch/pitch_checks/2026-09-24/kD_v1_svxy_movespike_b.py",
        "source": "stand_down",
        "expires": "2027-03-24",
    },
    {
        "added": "2026-09-24",
        "title": "Long TLT after a high-volume down day into a fresh 252-day low",
        "cell": "rates price-state, the down-day mirror of W16, volume x fresh low",
        "trigger": (
            "FILTERING, then a mechanism. TLT down 1.25% or more on 1.5x its 63-day volume, "
            "closing at a trailing-252 low, pays +0.643% at h=5 on 9-4 (13 declustered "
            "episodes) against own drift +0.021%, 25.7x cost, all 27 neighbour cells positive "
            "at h=5. It died because the at-the-low leg re-anchors rather than filters "
            "(filtering -0.690pp, re-anchoring +1.450pp at h=5; reanchor_null p 0.005), the "
            "MOVE-spike half the vol-targeting story names is the weak half (+0.136% on 4-3 "
            "against +1.235% on 5-1), and drop-top-2 leaves +0.027%. TURNS ON at a firing "
            "where ^MOVE rose less than +8.7% on the day, provided the kept-minus-dropped "
            "filtering term is >= 0 at h=5 on the pooled TLT+IEF form and a mechanism other "
            "than vol-targeting is written down BEFORE the check (the no-spike slice is "
            "post-hoc on 6 episodes)."
        ),
        "script": "scratch/pitch_checks/2026-09-24/kA_r1_tlt_voldown_b.py",
        "source": "stand_down",
        "expires": "2027-03-24",
    },
]

titles = {e["title"] for e in entries}
added = [e for e in new if e["title"] not in titles]
entries.extend(added)
w["asof"] = "2026-09-24"
path.write_text(json.dumps(w, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"appended {len(added)}; updated W28 and W18; active entries now {len(entries)}; expired {len(w.get('expired') or [])}")
