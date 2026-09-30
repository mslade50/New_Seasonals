import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

HERE = Path(__file__).resolve().parent
reg = ROOT / "data/pitch_negative_registry.md"
text = reg.read_text(encoding="utf-8")
block = (HERE / "_registry_append.md").read_text(encoding="utf-8")
if "## 2026-09-28:" not in text:
    reg.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")

w = load_watchlist()
entries = w["entries"]
titles = {e["title"] for e in entries}
new = [
    {
        "added": "2026-09-28",
        "title": "Short the dollar into payrolls from k=-4 with the ten-year at its 252-day high",
        "cell": "nfp x dollar_fx, the rates-high leg of the killed crowded-dollar run-in",
        "trigger": "OUT OF SAMPLE ONLY, and a remnant. The pre-specified cell (short DX k=-4 to the NFP close, gated on DX 21d rank >= 85 AND ^TNX within 1% of its 252 high) died: the DX-rank leg alone is wrong-signed (-0.083% on 24-26). The TNX-high leg alone pays short DX +0.317% on 8-2 at h=4 (sign p 0.055), UUP 6-0 (p 0.016), 4-0 since 2018, drop-best-2 +0.123%, against +0.096% for the same gate on non-NFP month turns. It is charged about p 0.3 for the 3-gate by 2-horizon walk, h=5 is 6-4, and the entry ladder is non-monotone (k=-3 4-6, k=-4 and k=-5 8-2). TURNS ON after 3 further out-of-sample NFPs with ^TNX within 1% of its 252 high at the k=-4 close score at least 2-1 at h=4 AND the k=-3 neighbour is positive on the pooled record. Today's firing (NFP 2026-10-02, entry the 09-28 close) is out-of-sample episode 1: score it on the 10-05 morning.",
        "script": "scratch/pitch_checks/2026-09-28/b2b_tnx_leg.py",
        "source": "stand_down",
        "expires": "2027-09-28",
    },
    {
        "added": "2026-09-28",
        "title": "Long gold 15% or more under its 252-day high with the dollar thrusting, above-200d half only",
        "cell": "gold x dollar_fx price-state",
        "trigger": "GLD BACK ABOVE ITS 200-DAY, plus a charge for the split. The pooled cell (GLD >= 15% under its 252 high AND DX-Y.NYB 21d rank >= 85) pays +1.132% at h=5 on 33-22 (sign p 0.088; GC=F 35-20, p 0.029), but the dollar gate re-anchors rather than filters (-0.43pp, reanchor p 0.015), and below the 200d the gate is worth +0.21pp at h=5 and negative at h=10, with 2018+ 7-7. The above-200d half is 7-0 at +4.02%, which is post-hoc. TURNS ON at a close with GLD above its 200d while the drawdown and DX legs both hold, provided the above-200d split is re-derived with the 2-regime by 24-cell walk charged. Today GLD is 5.53% under its 200d (393.41 against 416.44).",
        "script": "scratch/pitch_checks/2026-09-28/c5d_gld_200d_split.py",
        "source": "stand_down",
        "expires": "2027-03-28",
    },
]
for e in new:
    if e["title"] not in titles:
        entries.append(e)
for e in entries:
    if e["title"].startswith("Short the quarter's two best SPDRs"):
        note = ("2026-09-28 verdict: CHECK as a pre-read (w_55_window_dressing_rev.py). Ranked 10 sessions "
                "before quarter-end, 2018+ pays +0.688% on 23-11 (sign p 0.029) against the +0.40% bar, and "
                "September 2018+ is +2.545% on 7-1. Caveats: 2022 is +9.82 of the September total, ranking "
                "at the QE close makes September 2018+ wrong-signed (-0.548%, 4-4), and the walk charge is "
                "unpaid. The anchor is the 09-30 close, so the 09-30 morning owns the decision. Legs on the "
                "09-16 ranking: short XLE and XLV, long XLY and XLU.")
        if "2026-09-28" not in e.get("note", ""):
            e["note"] = (e.get("note", "") + " | " + note).strip(" |")
w["expired"] = []
save_watchlist(w)
print(len(entries), "entries")
