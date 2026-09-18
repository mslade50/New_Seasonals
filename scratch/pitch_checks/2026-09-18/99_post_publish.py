"""Post-publish housekeeping for 2026-09-18: registry append + watchlist update."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

REGISTRY = ROOT / "data" / "pitch_negative_registry.md"
TODAY = "2026-09-18"
DAY = "scratch/pitch_checks/2026-09-18"

SECTION = """

## 2026-09-18: quad witching after a post-FOMC crush, nine candidates, all empty

Nine candidates over four novelty axes and seven asset classes, three adversarial
checkers, stand-down. The 09-17 tape (FOMC k=+1): ^VIX -12.82% to 15.44
(VIX/VIX3M 0.832), SPY +1.13%, TLT +1.11% from 1.33% above its 252 low, GLD
+1.69% / SLV +3.37% / GDX +3.36% on a flat dollar, IWM z10 -1.10 into the
September quad (the event sleeve's T3 skip state), USO +18.9% over 21d.

### Method traps

- **V4's September 0-for-8 is a beta event, so September short-vol IS T3.**
  Raw short SVXY from the September opex close to +3 is 8-0 at +1.535% (sign
  p 0.004) post-break, but SPY fell -0.929% on the same windows (1-7) and the
  beta-charged alpha at 1.50 is +0.139% on 4-4 (sign p 0.637, 1.2x cost). The
  2026-08-06 routing note ("that stress is the T3 short-IWM trade") is now
  measured rather than asserted. Spot ^VIX residuals are not a stand-in for
  SVXY: the spot September cell reads 19-7 at h=2 while the 14-year ETP record
  is 7-7. (kA_c4_r1.py, kA_c4b_termstructure.py, kA_c4c_ladder_h12.py)
- **A vol crush into expiry is followed by risk-on continuation, not the
  vanna/charm reversal.** Short SPY from an opex close after ^VIX fell 10% into
  expiry pays -0.315% at h=2 on 19-31; the one-day crush form is 11-29 (t
  -3.16) for the short, and after a crush the ^VIX residual post-opex is
  -2.372% (13-27). The inverse (long SPY) is parked on the watchlist.
  (kA_c2_r1.py, kA_c2c_inverse_lead.py)
- **Confirmation buys speed, not size.** A complex-wide metals up day moves
  the continuation forward into D+1: SLV h=1 +0.498% at lag 0 (119-76) against
  -0.035% at lag 1. Magnitude-matched, confirmed days pay +0.578% at lag 0 and
  +0.275% at lag 1 h=3, while SLV-only days of the same size pay -0.533% and
  +0.904%. Always run magnitude-matched gate attribution at the tradeable lag.
  (kC_c6_r1.py, kC_c6c_magnitude.py)
- **Check the beta sign before reading a point-in-time-beta underreaction
  gate.** ITB on TLT ran about -1.5 from 2008 to 2020 and +1.5 now, so the
  same gate selects different days in each regime; split by beta sign first.
  (kC_c7_r1.py, kC_c7b_exposure.py)
- **Split a band into its halves before calling it the inversion of its
  neighbour.** TLT's [1.0,1.5) thrust band is all [1.00,1.25) (+0.394%, 15-8)
  with [1.25,1.50) at -0.058%. (kC_c8_r1.py)
- **A pooled event cell owes its worst month its own estimate.** Before
  trusting a pooled quad cell in September, add the gate's month-demeaned lift
  to that month's ungated mean: the washout gate's +1.124pp over 40 opexes on
  September's -1.602% predicts -0.48%, while the quad-only lift (11 episodes)
  predicts +1.09%. When the two disagree in sign the live month is not
  covered. (kB_c1d_sept.py)
- **For China holidays, residualise Hong Kong against EEM on the same dates.**
  The 2015+ "into Golden Week" FXI weakness ex-2024 (-1.296%) is ordinary
  late-September EM weakness (EEM -1.298% on the same windows). Golden Week's
  T-1 is always the quarter's last session. Lunar New Year closures can be
  dated from ^HSI's own gaps in the cache. (kB_c10_r1.py, kB_c10b_hsi.py)

### Cells swept and empty

- **Long IWM from the September quad-witching close after a small-cap washout
  (IWM z10 <= -1).** Pooled across quads 10-1 at h=8 (+3.655%, sign p 0.006),
  ex-September 9-1; the live month is uncovered (above) and its ungated
  post-quad windows run 6-19 at -2.03%; washouts on any day from September
  10-25 go 4-8. Parked for non-September quads. (kB_c1_r1.py, kB_c1b_gate.py,
  kB_c1c_live.py, kB_c1d_sept.py)
- **Short SPY from a monthly opex close after a VIX crush into expiry.** Above;
  placebo 5, 8 and 5 of 11; option positioning history (90 rows, all
  2026-08-05) cannot test dealer hedging. (kA_c2_r1.py)
- **SPY-hedged short SVXY the session after a >= 12% VIX crush (watchlist 51
  live).** Parent +0.326% at h=1 on 62-40, 2.7x cost; the [12,15) bucket
  -0.073% (20-21); entry on an opex close 2-3 at h=1, 1-4 at h=3; FOMC k=+1
  crushes +0.116% against +0.321% without. The dose ordering flips across the
  2018 leverage break. (kA_c3_r1.py)
- **Hedged short SVXY over the September post-opex window.** Above.
- **Long SLV after a complex-wide metals up day.** Above; deep-drawdown row 7
  episodes since 2018, all 2026, -0.383%; energy and rates complexes show no
  family effect (XOP -0.456pp at h=3). (kC_c6_r1.py, kC_c6b_refclass.py)
- **Long ITB the session after a bond rally it underreacted to.** The catch-up
  session is -0.696% on 18-32; the 2022+ cross-sector rank correlation of TLT
  beta with next-day excess is -0.03; SPY-hedged 2022+ h=5 -0.231%.
  (kC_c7_r1.py, kC_c7b_exposure.py)
- **Long TLT after a +1.0% to +1.5% session within 4% of its 252 low.** Band
  above; signed top two 91% of the h=2 total; charged over the 125-cell grid
  P = 1.000 at the selected cell's |t| of 0.869; within 2% of the low h=5
  -0.810%. (kC_c8_r1.py, kC_c8b_search.py)
- **Short crude from September tdom 13 into the refinery-turnaround season
  after a 21d thrust.** USO 2015-2025 ranks the September short 6 of 12 at h=10
  (-0.17%), CL=F 6 of 12 (-0.20%); September thrust episodes 0-4 for the short
  at h=5; today's entry 11 of 11 on the thrust ladder. The trough is October
  (CL=F October tdom-13 short rank 1 in 2001-2014), parked. (kB_c9_r1.py)
- **Short FXI into and across the Golden Week Stock Connect suspension.** 2015+
  run-in residual vs EEM +1.601% (6-5); across the closure FXI rises 16-5;
  offset ladder 8 of 11; Golden Week plus Lunar New Year +0.153% on 11-12.
  (kB_c10_r1.py, kB_c10b_hsi.py)
"""

NEW_ENTRIES = [
    {
        "added": TODAY,
        "title": "Long IWM from a quad-witching close after a small-cap washout, outside September",
        "cell": "quad_witching x us_small, washout gate (inversion of the event sleeve T3 skip rule)",
        "trigger": ("A DATE AND A LEVEL. Pooled across quarterly quads, IWM z10 (event-sleeve definition: "
                    "10-session return over vol21*sqrt(10)) <= -1.0 at the prior close pays +3.655% at h=8 "
                    "on 10-1 (sign p 0.006) against a -0.003% discarded complement, ranks 1 of 11 on the "
                    "offset ladder at h=5/8/10, holds 4-1 before 2018 and 6-0 after, 7-0 with no FOMC in "
                    "the hold, drop-best-2 +2.70%. Ex-September 9-1 at +3.116% (sign p 0.0107) against a "
                    "+0.742% complement; December rows 4-0. TURNS ON at a non-September quad (next "
                    "2026-12-18, signal 12-17) with IWM z10 <= -1.0. It died on 2026-09-18 only because "
                    "September is uncovered: one gated September (2001 reopening), ungated September "
                    "post-quad 6-19 at -2.03%, month-demeaned lift predicts -0.48%. Dose note: the all-opex "
                    "(-1.0,-0.75] bin is the worst on every z definition (h=8 -0.334%), so a reading just "
                    "under -1.0 sits beside a cliff; report the live bin. Owes round 3 (horizon, entry "
                    "form, loser paths) when it fires."),
        "script": f"{DAY}/kB_c1d_sept.py",
        "source": "stand_down",
        "expires": "2026-12-21",
    },
    {
        "added": TODAY,
        "title": "Long SPY from a monthly opex close after a one-day VIX crush in the three sessions before",
        "cell": "opex x volatility state -> us_large, the inversion of the killed vanna short",
        "trigger": ("THE CYCLE YEAR, and it owes a search charge. A >= 10% one-day ^VIX fall in the three "
                    "sessions before a monthly opex, long SPY MOC on the opex close, pays +0.535% at h=2 "
                    "over 40 anchors at 29-11 (sign p 0.003) against +0.040% for the same crush on "
                    "non-opex days and +0.091% for all opex closes, placebo rank 1 of 11, 18-9 before "
                    "2018 and 11-2 after. It was the best of about 12 cells (charged p about 0.04). "
                    "Midterm years 4-5 at +0.099% against 25-6 at +0.661% otherwise; September 2-1 at "
                    "+0.060%. TURNS ON at a non-midterm, non-September opex (first chance 2027-01-15) "
                    "after a >= 10% one-day crush in the three prior sessions, with a fresh placebo "
                    "ladder and the midterm split re-run on the day."),
        "script": f"{DAY}/kA_c2c_inverse_lead.py",
        "source": "stand_down",
        "expires": "2027-09-17",
    },
    {
        "added": TODAY,
        "title": "Short CL=F from October trading day 13 into the refinery-turnaround trough",
        "cell": "energy x month-of-year, fall refinery maintenance",
        "trigger": ("A DATE, then the roll. On the 2026-09-18 month ladder the October tdom-13 short ranks "
                    "1 of 12 on CL=F 2001-2014 (+1.90% at h=10) and 2 of 12 on USO 2015-2025 (+1.48%), "
                    "about 15-10, while the September boundary ranks 6 of 12. Found on a ladder, so it "
                    "is charged for the 12-month walk. TURNS ON at the 2026-10-19 close if that "
                    "morning's script reproduces a top-3 rank in both eras, a front-contract form that "
                    "handles the November roll inside the window clears 5x cost, and the USO form "
                    "(+0.95% on 11-9) is reported beside it."),
        "script": f"{DAY}/kB_c9_r1.py",
        "source": "stand_down",
        "expires": "2026-10-20",
    },
]

NOTES = {
    "Beta-hedged short SVXY the session after a 10% one-day VIX crush, 2018+": (
        "2026-09-18 verdict: CHECK, killed. State fired (^VIX -12.82% on 09-17). Parent re-runs at "
        "+0.326% at h=1 on 62-40 (2.7x cost); the [12,15) bucket -0.073% (20-21); entry on an opex "
        "close 2-3 at h=1; FOMC k=+1 crushes +0.116% vs +0.321%. The dose ordering flips across the "
        "2018 leverage break (synthetic [10,12) -0.017% vs >=12% +0.367%), so the dose leg of the arm "
        "is noise; the COST leg (+0.60% at h=1) is the binding condition. (kA_c3_r1.py)"),
    "Short TLT after a big up day from inside the 52-week low zone": (
        "2026-09-18 verdict: PASS on the short (TLT +1.11%). The long-side read of the [1.0,1.5) band "
        "was checked and killed: the whole band is [1.00,1.25) (+0.394% to the long, 15-8), "
        "[1.25,1.50) -0.058%, top two episodes 91%. (kC_c8_r1.py)"),
    "Short silver after the whole metals complex breaks together": (
        "2026-09-18 verdict: PASS (complex rose). The UP-day mirror was checked and killed: the "
        "continuation lands on D+1 (lag 0 +0.498% vs lag 1 -0.035% at h=1), and confirmed days pay "
        "less than SLV-only days at the tradeable lag. (kC_c6_r1.py, kC_c6c_magnitude.py)"),
}


def main() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    marker = "## 2026-09-18: quad witching after a post-FOMC crush"
    if marker in text:
        print("registry: section already present, skipped")
    else:
        REGISTRY.write_text(text.rstrip("\n") + SECTION, encoding="utf-8")
        print("registry: appended 2026-09-18 section")

    w = load_watchlist()
    entries = w["entries"] if isinstance(w, dict) else w
    titles = {e.get("title") for e in entries}
    added = 0
    for e in NEW_ENTRIES:
        if e["title"] not in titles:
            entries.append(e)
            added += 1
    noted = 0
    for e in entries:
        note = NOTES.get(e.get("title"))
        if note and TODAY not in e.get("note", ""):
            e["note"] = (e.get("note", "") + " | " + note).strip(" |")
            noted += 1
    kept = [e for e in entries if str(e.get("expires", "9999")) >= TODAY]
    pruned = len(entries) - len(kept)
    if isinstance(w, dict):
        w["entries"] = kept
    else:
        w = kept
    save_watchlist(w)
    print(f"watchlist: +{added} entries, {noted} notes, {pruned} pruned, {len(kept)} active")


if __name__ == "__main__":
    main()
