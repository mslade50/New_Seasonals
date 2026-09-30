import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "data" / "pitch_watchlist.json"
wl = json.loads(path.read_text(encoding="utf-8"))
entries = wl["entries"]

before = len(entries)
entries = [e for e in entries
           if not e["title"].startswith("Short the quarter's two best SPDRs against its two worst")]
assert len(entries) == before - 1, "W55 not found exactly once"

for e in entries:
    if e["title"].startswith("Long UNG for two sessions when the volume thrust re-fires"):
        e["trigger"] += (" SCORED 2026-09-30: the 2026-09-24 firing (entry 09-25 close 11.13, exit "
                         "09-29 close 10.35) lost -7.0% at h=2. Out-of-sample record 0-1.")

new = [
    {
        "added": "2026-09-30",
        "title": "Long gold from the quarter-end close after a quarter-end-week flush, yields not rising",
        "cell": "quarter-end close x gold_miners x rates regime",
        "trigger": ("THE YIELD LEG, and a charge for the split. GLD 5d return <= -2% at the QE-1 close, "
                    "long GLD from the quarter-end close for 7 sessions, pays 10-0 at h=7 (p 0.001); the "
                    "pre-specified QE-close -3% form is 7-0 at +3.58% (p 0.008), QE+0 ranks 1 of 13 on "
                    "the anchor ladder, and the same flush on ordinary days pays +0.70%. It died on "
                    "2026-09-30 because the rebound only exists when yields FELL into the close: with the "
                    "^TNX 5-day change > 0 it pays +0.25% on 3-3 at h=5 (drift), with it <= 0 +2.25% on 9-1 "
                    "and +3.73% on 10-0 at h=7, and the one flush with ^TNX at its 252 high (2023-09-29) "
                    "lost 1.02% at h=5. Today ^TNX was +28.7 bp. TURNS ON at a quarter-end where GLD's "
                    "5-day return at the QE-1 close is <= -2% AND the ^TNX 5-day change into the QE-1 close "
                    "is <= 0; score it out of sample (the yield split was found inside this check). Two "
                    "standing debts: gold is not sold into quarter-ends on average (+0.37% run-in against "
                    "+0.23% all days), and the rule is wrong-signed on SLV (6-6) and GDX (4-5), so the "
                    "mechanism is unexplained. Next look 2026-12-31. Vehicle GLD or GCZ6/MGCZ6."),
        "script": "scratch/pitch_checks/2026-09-30/kC_c6c_gld_qe_regime.py",
        "source": "stand_down",
        "expires": "2027-07-15",
    },
    {
        "added": "2026-09-30",
        "title": "Quarter-end sector reversal at high dispersion, ex-XLE (replaces watchlist 55)",
        "cell": "quarter-end close x us_large sectors",
        "trigger": ("THE EX-ENERGY DISPERSION. W55 fired on 2026-09-30 and died: 2018+ +1.010% on 22-12, "
                    "but 2021 and 2026 are 85% of it, the QE label is worth -0.140pp before 2018, QE+1 is "
                    "42-62, and removing XLE takes the whole pair to +0.013%. What is left is dispersion: "
                    "top-2 minus bottom-2 eight-SPDR (ex-XLE) 63d gap at or above its p80 (18.2pp), short "
                    "the two best and long the two worst from the QE close to QE+5, pays +0.644% on 14-7, "
                    "+1.353% since 2018. TURNS ON at a quarter-end where that ex-XLE gap is >= 18.2pp; on "
                    "2026-09-29 it was 15.2pp (22.4pp with XLE). Standing debts: QE+1 is still negative at "
                    "high dispersion (-0.160% at p80), September at p80 is 2-2, and the dispersion gate "
                    "came from this morning's walk (charged p 0.054 at p50). Next look 2026-12-31."),
        "script": "scratch/pitch_checks/2026-09-30/kB_c1d_disp_form.py",
        "source": "stand_down",
        "expires": "2027-07-15",
    },
    {
        "added": "2026-09-30",
        "title": "Long crude on the FIRST -3% day after a 21-day thrust, CL=F front contract",
        "cell": "price state x energy",
        "trigger": ("A PRE-REGISTERED FORWARD TEST. The 2026-09-28 short (first -3% USO day with 21d rank "
                    ">= 75) was wrong-signed in all 28 cells; its long side holds WITHOUT 2026 at +1.25% "
                    "at h=3 on 16-3 (sign p 0.002), the first crude-drop long not carried by one regime. "
                    "The 63d-rank neighbour tested 2026-09-30 is a 2026 artifact (ex-2026 +0.48% on 10-8, "
                    "-3.5 to -4.0% falls -2.44% on 3-14), so the definition is the 21d form only. This is a "
                    "sign flip recovered inside a kill, so it owes forward evidence. REGISTERED TODAY: "
                    "signal = the first USO close down >= 3% with no other >= 3% down close in the prior 8 "
                    "sessions, USO 21d rank >= 75 (trailing-252 pctile) on the signal close; long CL=F "
                    "front (or USO) from the next close, h=3, lag 1, no stop. TURNS ON when the first 5 "
                    "firings from 2026-09-30 onward go 4-1 or better at a mean of +1.0% or more; check "
                    "that no payrolls session is inside the hold (that slice paid -0.54% on 3-3 on the 63d "
                    "form). Merge with watchlist 63 (round-trip long) when either scores."),
        "script": "scratch/pitch_checks/2026-09-30/kC_c9c_family_and_book.py",
        "source": "stand_down",
        "expires": "2027-09-30",
    },
]
entries.extend(new)
wl["entries"] = entries
wl["expired"] = []
wl["asof"] = "2026-09-30"
path.write_text(json.dumps(wl, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"entries {before} -> {len(entries)}")
