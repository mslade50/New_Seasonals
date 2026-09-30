"""Post-publish: append 2026-09-23 near-misses to data/pitch_watchlist.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "data" / "pitch_watchlist.json"
w = json.loads(path.read_text(encoding="utf-8"))

new = [
    {
        "added": "2026-09-23",
        "title": "Long EEM against beta-SPY from QE-5 into the December quarter-end close",
        "cell": "international x quarter-end, the long residual flip of the killed funding short",
        "trigger": "A DATE, then a flip charge. Found as the opposite sign of the 2026-09-23 EEM quarter-end short (which lost -0.803% on 33-60). Long EEM against 0.93-beta SPY from the QE-5 close to the quarter-end close pays +0.63% at t 4.04 (59-34) over 93 quarters, December 18-5, September 10-13, and it has decayed (pre-2013 +1.34%, 2013+ +0.42%). The dollar leg of the funding story is flat on the same anchors, so no mechanism is attached yet. TURNS ON at the 2026-12-23 QE-5 close if the 2013+ December row alone clears 5x a two-leg round trip after charging the sign flip (2 signs x the horizons scanned), and a mechanism other than dollar funding is written down first.",
        "script": "scratch/pitch_checks/2026-09-23/kA_a3_eem_qe.py",
        "source": "short_slate",
        "expires": "2026-12-31",
    },
    {
        "added": "2026-09-23",
        "title": "Long crude after a 21-day thrust round-trips inside five sessions",
        "cell": "energy price-state, thrust x flush, the long flip of the killed round-trip short",
        "trigger": "A PRE-REGISTERED FRONT-CONTRACT FORM at the next firing. USO r5 <= 3 within 10 sessions of r21 >= 90 pays the LONG +1.82% at h=5 on 9-5 (sign p 0.21), and the prior thrust is worth +3.29pp against the same flush without one (which pays the short +1.47% on 54); the short is negative in 27 of 27 USO cells and positive in 1 of 27 CL=F cells. It is a post-hoc sign flip on 14 episodes and CL=F is 12-10. TURNS ON when a CL=F front-contract long, registered before the check with its horizon fixed at h=5, clears sign p 0.05 on the round-trip state; live on 2026-09-22 (USO r5 2.38, r21 90.87 on 09-15, CL=F r5 5.2).",
        "script": "scratch/pitch_checks/2026-09-23/kC_b3_uso_roundtrip_b.py",
        "source": "short_slate",
        "expires": "2026-10-14",
    },
]

titles = {e["title"] for e in w["entries"]}
added = [e for e in new if e["title"] not in titles]
w["entries"].extend(added)
w["asof"] = "2026-09-23"
path.write_text(json.dumps(w, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"appended {len(added)}; active entries now {len(w['entries'])}")
