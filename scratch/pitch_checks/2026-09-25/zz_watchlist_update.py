"""Post-publish: 2026-09-25 near-misses and verdict notes."""
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


tlt = find("Long TLT after a high-volume down day into a fresh 252-day low")
add_note(tlt, "2026-09-25 verdict: PASS. The state fired again on 09-24 (TLT -1.29% on 1.94x "
              "at a fresh 252 low) but ^MOVE rose +9.57% against the < +8.7% arm leg.")

spike = find("Long SVXY hedged against SPY after a bond-vol spike below the extreme tail")
add_note(spike, "2026-09-25 verdict: PASS. ^MOVE +9.57% is the 97.4th pctile of daily moves "
                "(just above the 90-97 band) and SPY -0.08% fails the fell-more-than-0.75% leg.")

new = [
    {
        "added": "2026-09-25",
        "title": "Long the peso for two sessions after a 1.5% carry-unwind day",
        "cell": "dollar_fx carry currencies, one-day unwind x cross-asset vol, the first carry cell in the repo",
        "trigger": (
            "A FRESH FIRING AT THE PRE-REGISTERED RUNG, h=2 ONLY. The pitched cell (USDMXN >= "
            "+1.25% with ^MOVE or ^VIX up) died on concentration (122 episodes, +0.130% at h=2, "
            "62-60, top two 61%) and a flat horizon profile. The >= +1.50% rung on the same "
            "rule pays +0.499% at h=2 spot over 82 episodes, 47-35, t 2.95, drop-best-2 "
            "+0.390%, top two 24%, pre/post-2018 +0.516% / +0.472%, but it was found on a "
            "threshold ladder of about 48 cells, the hit rate is 57% so the mean is skew, and "
            "it fades to +0.275% at h=5 (pre-2018 -0.006%). 2026-09-24 printed +1.479%, 2 bp "
            "short. The rung is REGISTERED TODAY at exactly +1.50%, h=2, lag 1, spot plus "
            "carry, MXN futures (6M). TURNS ON at the next USDMXN session >= +1.50% with ^MOVE "
            "or ^VIX up on the day, provided no NFP falls inside the h=2 hold (NFP-in-hold "
            "episodes run -0.336% at h=5). Score each firing out of sample; do not re-walk "
            "the ladder."
        ),
        "script": "scratch/pitch_checks/2026-09-25/kC_x1_mxn_carry_c.py",
        "source": "stand_down",
        "expires": "2027-03-25",
    },
    {
        "added": "2026-09-25",
        "title": "Long UNG for two sessions when the volume thrust re-fires with no storage report inside the hold",
        "cell": "energy price-state, the re-fire population of the 2026-09-23 pitched UNG rule",
        "trigger": (
            "OUT OF SAMPLE ONLY. Re-fires of the >= +5% / >= 3x 63d volume rule within five "
            "sessions of a prior firing pay +0.84% at h=2 on 6-4 (sign p 0.377) against +4.26% "
            "on 14-2 for first firings, and inside the re-fire state the volume gate does not "
            "separate (+5% days without it +1.41%). The slice with no Thursday EIA report in "
            "the h=2 hold is 6-1 at +3.66% (sign p 0.0625), and today's shape (no Thursday, "
            "run-in >= 3%) is 5-1 at +2.10%, but that is a post-hoc split of a post-hoc split "
            "and 2011+ it is 3-1 at +0.69%. TURNS ON when the first 5 re-fires from 2026-09-25 "
            "onward with no Thursday inside the hold go 4-1 or better with a mean of +1.5% or "
            "more at h=2 lag 1. The 2026-09-24 firing (entry 09-25 close, exit 09-29 close) is "
            "the first to score."
        ),
        "script": "scratch/pitch_checks/2026-09-25/kA_n1_ung_refire_b.py",
        "source": "stand_down",
        "expires": "2027-09-25",
    },
]

titles = {e["title"] for e in entries}
added = [e for e in new if e["title"] not in titles]
entries.extend(added)
w["asof"] = "2026-09-25"
path.write_text(json.dumps(w, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"appended {len(added)}; noted W64 and W65; active entries now {len(entries)}; "
      f"expired {len(w.get('expired') or [])}")
