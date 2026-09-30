"""Post-publish housekeeping for 2026-09-21: registry append + watchlist update."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

REGISTRY = ROOT / "data" / "pitch_negative_registry.md"
TODAY = "2026-09-21"
DAY = "scratch/pitch_checks/2026-09-21"

SECTION = """

## 2026-09-21: a rate shock under a calm index, eleven candidates in two waves, all empty

Eleven candidates over five novelty axes and six asset classes, four adversarial
checkers, stand-down. The 09-18 tape (quad witching): ^TNX 4.998 against a 252
max of 5.006 (+34.5 bp in 21 sessions), IEF 0.08% and LQD 0.40% above their 252
lows, ^MOVE +5.80%, DX 5d rank 91, USO +17.5% over 21d, XLU at its 52w low with
17 other tape names within 1% of one, banks 11 of 11 at r5 <= 20, SPY 1.84% off
its high with ^VIX 14.81 (VIX/VIX3M 0.812). Watchlist 23 fired for the first time
and was retired.

### Method traps

- **An armed arm that re-anchors is a delay rule, not a filter, and the arm text
  should say which it is before it fires.** W23 (XLU r21 <= 5 AND TLT r21 < 25)
  reproduced exactly on firing (+0.858% at h=5, 21-7) and still died: at the
  same anchors the TLT gate is worth -0.085pp (discarded washouts +0.339%,
  kept +0.071%), 90-96% of the edge is the later entry date (reanchor null p
  0.012), and the rate-sensitive family on the identical form is wrong-signed
  (XLRE -0.924pp, XLP -0.691pp, ITB -2.097pp, XHB -0.915pp). Utilities are now
  dead in nine expressions. (kA_c1_r1.py, kA_c1_r2.py)
- **Check ^IRX before calling a ten-year thrust a steepener.** The ten-year rose
  +34.5 bp in 21 sessions while the 10y-3m spread widened only +6.7 bp and
  10y-5y narrowed 15.8 bp: a bear parallel shift. (kA_c5_r1.py)
- **A round-number level needs off-round placebo levels, and on ^TNX they win.**
  The whole-percent cell is the worst of the four quarter grids at h=3 and h=5
  (-0.393% on 2-3 against .25 +2.146%, .50 +0.322%, .75 -0.014%). The .25 grid's
  6-0 is a placebo byproduct and is not parked. (kA_c2_r1.py, kA_c2_r2.py)
- **Liquid laggards RISE into their prints.** The short-the-laggard pre-print
  cell loses at every hold 1-7 (-0.93% beta-hedged on 69-100 week clusters,
  -1.58% since 2018), and same-week no-print pairing leaves +0.17pp (t 0.52).
  The 2026-09 COST "keeps falling into its print" finding is one name and does
  not generalize. The earnings offset ladder failed three more anchors today
  (12 of 16, 12 of 16, 6 of 16). (kB_c3_r1.py, kB_c9_r1.py)
- **An expiry-day gap in an international ETF reverses on the NEXT session,
  which a lag-1 MOC cannot reach.** EFA after an opex lag of >= 1pp vs
  beta-SPY: lag 0 10-1 (+0.450%), lag 1 2-9 (-0.585%). MSCI EAFE reviews
  rebalance at the end of Feb/May/Aug/Nov, not at quad witching. (kC_c6_r1.py)
- **The VX future that prices a Nov 3 election is OCTOBER's.** It settles about
  10-21 and its 30-day window spans the vote, so short-vol ETPs already hold
  the election kink in late September; there is no October roll INTO it.
  (kC_c7_r1.py)
- **A duration-hedged LQD spread proxy is half equity beta over multi-week
  windows** (R-squared 0.50 on SPY's window return); neutralize on ex-ante IEF
  and SPY betas before reading any IG calendar cell. (kD_c11_r2.py)

### Cells swept and empty

- **Long XLU with the long end hit alongside (watchlist 23, armed).** Above.
  Definition neighbours: 10d lookback -0.204pp, 42d -0.188pp, IEF in place of
  TLT +0.176pp; the opposite gate (TLT r21 > 75) +0.849pp on 12. Midterm with
  SPY above its 200d 3-1 on N=4, so the W35 adverse slice did not bite here.
- **Long TLT at a whole-percent yield after a thrust.** Above; episodes 2006-04
  5%, 2013-09 3%, 2022-05 3%, 2022-09 4%, 2023-10 5%; midterm 0-3 (-1.27% h=5).
- **Long XLF on an 11-bank breadth floor under a bear steepener.** Not live
  (above); the curve gate is worth -0.134pp at h=5, 2008-10-08 and 2009-03-04
  are 62% of the total, and midterm-above-200d is 0-3 (-1.19%).
- **Short a 52w-low laggard into its print (NKE live).** Above; PCG 2019-10-28
  (+81% in the window) is the tail; NKE's own gated record 1-2.
- **Long a 63d winner into its print (GIS, PAYX, CAG live).** The r63 >= 80
  gate subtracts at both forms (-0.027% vs +0.017% complement; CAG form
  -0.11pp, r63 < 20 best at +0.32%). The r63 >= 80 AND r21 <= 15 cross
  (+0.82% on 24-11) is parked, see the watchlist. (kB_c9b_cross.py)
- **Single-stock window dressing into the quarter-end.** QE-7 to QE -0.16%
  beta-hedged vs -0.58% at ordinary month-ends (Welch t 0.82); December 6-13;
  QE to QE+5 reversal -0.10%. The liquid-name September row (+1.04%, 15-4) is
  late-September weakness in weak names, not a quarter-end object: the true
  QE end-date ranks 4 of 21 and the row does not replicate outside the liquid
  set (+0.08%). (kB_c4_r1.py, kB_c4b_sep.py)
- **Long EFA against beta-SPY after an expiry-day lag.** Above. The
  unconditioned byproduct (EFA vs beta-SPY after quads, +0.329% at h=3 on
  63-35, quad date 1 of 11 on the ladder) has faded to 17-17 (+0.105%, 1.3x
  cost) since 2018. (kC_c6b_parent.py)
- **Short SVXY across an election-year October.** SPY-adjusted 4-3 (+1.39%,
  sign p 0.50) vs odd years +0.06%; the near-low VIX gate fired once (2012,
  -1.12%); every entry offset -5..+5 is positive, which is the SPY tape. It
  would also fight V4 (long SVXY 10-16 to 10-21). (kC_c7_r1.py, kC_c7b_round2.py)
- **Long yen into Japan's fiscal half-year and year-end.** March plus September
  18-35 (-0.211%, sign p 0.994) against June/December -0.053% and month-ends
  +0.046%; September 2017+ 1-8. The inverse (short yen) is a spring drift that
  ranks 9 of 11 on its own ladder and duplicates the long-DX ship. (kC_c8_r1.py)
- **Watchlist 56's unmeasured arm, measured.** Long DX from QE-9 to QE in
  quarters with no FOMC decision in the window, 2008+: +0.209% on 23-18,
  +0.197pp over ordinary month-ends (t +0.94), +0.021pp if a decision on the
  entry day counts. Below its +0.25pp bar. (kC_c8_r1.py)
- **Long SPY from the late-September close into a midterm election.** Midterms
  +2.75% on 5-1 to the pre-election close, but odd years +3.60% on 11-2 and
  presidential years -3.09% on 3-4; above-200d midterms 3-1 at +0.82% under the
  +1.13% drift; live rung 12 of 21 on the offset ladder, September-midterm 23 of
  48 on the month x cycle grid, first 10 sessions 1-5. The post-election anchor
  (+0.89%, 4-2) does not carry it either. (kD_c10_r1.py)
- **Long LQD against beta-IEF into the pre-earnings issuance blackout.** Pooled
  +48.3 bps at h=17 on 63-30 (ladder 3 of 68), but September is the only losing
  quarter (-24.7 bps on 12-11, permutation P 0.011; HYG -71.5 bps) and the
  equity-neutral pooled cell is +18.9 bps (sign p 0.107). The December row is
  parked. (kD_c11_r1.py, kD_c11_r2.py)
"""

W23_TITLE = ("The utilities washout with the long end hit ALONGSIDE it, which is the negation "
             "of the cell pitched on 2026-08-25")

NEW_ENTRIES = [
    {
        "added": TODAY,
        "title": "Long a 63-day winner that has lagged over 21 days, into its earnings print (the GIS form)",
        "cell": "earnings print x price state (r63 >= 80 AND r21 <= 15), single names",
        "trigger": ("SPY BELOW ITS 200-DAY at the signal close, plus a search charge that has never been "
                    "paid. Long the name MOC two sessions before its announcement date, exit at the "
                    "pre-print close (k=2, h=1), beta-hedged: +0.82% on 24-11 over 35 liquid events "
                    "(sign p 0.020) against +0.14% for the same cross with no print, true anchor 1 of 16 "
                    "on the offset ladder in both universes, monotone in the 21-day rung (<= 5 +2.06%, "
                    "<= 10 +0.88%, <= 20 +0.48%, <= 25 +0.30%). It was 1 of 16 cross rows tried inside a "
                    "kill, so it is charged. With SPY above its 200d (2026-09-21) it pays +0.24% on 16-10, "
                    "2.4x a 10 bp round trip; below the 200d 8-1 at +2.50%, where 2002 and 2008 carry "
                    "43%. 2018+ +0.25% on 9-6, and starting at k=3 flips the broad universe to -0.04%. "
                    "TURNS ON when a liquid name in the cross has a print within 2 sessions while SPY "
                    "closes below its 200d, with a fresh broad-universe replication and the k=3 row "
                    "re-run on the day."),
        "script": f"{DAY}/kB_c9b_cross.py",
        "source": "stand_down",
        "expires": "2027-03-19",
    },
    {
        "added": TODAY,
        "title": "Long LQD against beta-IEF from seven sessions before the December quarter-end",
        "cell": "credit x quarter-end, December only",
        "trigger": ("A DATE, then an equity-neutral bar. The December QE-7 row of the killed "
                    "issuance-blackout cell pays +45 bps at h=17 on 19-4, equity-neutral on ex-ante IEF "
                    "and SPY betas +29 bps on 15-8, against -17 bps at ordinary month-ends. Its return "
                    "sits before the quarter-end (+36 of +45 bps), so it reads as a year-end effect and "
                    "the supply-drought mechanism does not apply (September is the only losing quarter, "
                    "-24.7 bps on 12-11). It owes its own mechanism. TURNS ON at the 2026-12-21 close if "
                    "the equity-neutral December form beats ordinary month-ends by >= 25 bps on a fresh "
                    "run, with the HYG row and a December offset ladder reported beside it."),
        "script": f"{DAY}/kD_c11_r2.py",
        "source": "stand_down",
        "expires": "2026-12-22",
    },
    {
        "added": TODAY,
        "title": "The utilities washout with TLT hit alongside, as a DELAY rule (retired watchlist 23)",
        "cell": "us_large sector x rates state",
        "trigger": ("OUT OF SAMPLE ONLY. Watchlist 23 fired for the first time on 2026-09-18 (XLU r21 "
                    "1.59, TLT r21 24.60) and was killed: the cell reproduces (+0.858% at h=5, 21-7, sign "
                    "p 0.006) but the TLT gate is worth -0.085pp at the same anchors, 90-96% of the edge "
                    "is the later entry date (reanchor null p 0.012), the rate-sensitive family on the "
                    "same form is wrong-signed, and the 10d and 42d lookbacks are negative. Only the "
                    "timing claim survives, and it has no mechanism. TURNS ON only if the joint state's "
                    "forward firings from 2026-09-21 (first day of each cluster, h=5) beat XLU's own "
                    "r21 <= 5 washout on the same dates by >= +0.25pp over at least five new episodes. "
                    "The 2026-09-18 firing is the first of them."),
        "script": f"{DAY}/kA_c1_r2.py",
        "source": "stand_down",
        "expires": "2027-09-21",
    },
]

NOTES = {
    "Long the dollar into the September quarter-end close": (
        "2026-09-21: arm MEASURED and off. No-FOMC quarters since 2008, QE-9 to QE long DX +0.209% on "
        "23-18, +0.197pp over ordinary month-ends (t +0.94); +0.021pp if a decision on the entry day "
        "counts. Below the +0.25pp bar. (kC_c8_r1.py)"),
    "Short regional banks against the big-bank index on a breadth washout": (
        "2026-09-21 verdict: PASS (breadth 11 of 11, median r63 34.5). The rates-gated long-XLF form "
        "(bear steepener) was checked and killed: not live (10y-3m +6.7 bp vs 25 bp) and the curve "
        "gate is worth -0.134pp. (kA_c5_r1.py)"),
}


def main() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    marker = "## 2026-09-21: a rate shock under a calm index"
    if marker in text:
        print("registry: section already present, skipped")
    else:
        REGISTRY.write_text(text.rstrip("\n") + SECTION, encoding="utf-8")
        print("registry: appended 2026-09-21 section")

    w = load_watchlist()
    entries = w["entries"] if isinstance(w, dict) else w
    before = len(entries)
    entries = [e for e in entries if e.get("title") != W23_TITLE]
    retired = before - len(entries)
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
            e["note"] = ((e.get("note") or "") + " | " + note).strip(" |")
            noted += 1
    kept = [e for e in entries if str(e.get("expires", "9999")) >= TODAY]
    pruned = len(entries) - len(kept)
    if isinstance(w, dict):
        w["entries"] = kept
    else:
        w = kept
    save_watchlist(w)
    print(f"watchlist: -{retired} retired, +{added} entries, {noted} notes, {pruned} pruned, "
          f"{len(kept)} active")


if __name__ == "__main__":
    main()
