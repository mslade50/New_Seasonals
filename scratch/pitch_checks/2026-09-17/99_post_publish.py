"""Post-publish housekeeping for 2026-09-17: registry append + watchlist update."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

REGISTRY = ROOT / "data" / "pitch_negative_registry.md"
TODAY = "2026-09-17"

SECTION = """

## 2026-09-17: four unopened calendar anchors and three relative-value states, all empty

Eight candidates over four novelty axes and six asset classes, three adversarial
checkers plus an independent rerun of the relative-value set, stand-down. The
09-16 tape (the FOMC decision session): ^TNX closed at a fresh trailing-252 max
(5.006) while TLT rose +0.21% and IEF fell (belly-led), DX-Y.NYB 5d rank 98.0,
USO -3.52% off a +19.9% 21d thrust while VLO printed a 252 high, banks broke
intraday on a -0.44% index (USB -2.35 ATR, PNC -2.14, GS -1.43), and 15 tape
names sat within 1% of a 52w low with SPY 3.06% off its high.

### Method traps

- **A checker's numbers are not evidence until its scripts are on disk.** One
  checker returned a full report whose cited scripts did not exist yet; its
  later script-backed report disagreed on the decisive live reading (the
  GDX-minus-GLD percentile 16.3 against 17.06, the 252 low -9.03pp against
  -10.37pp). Every number in a checker report must trace to a saved script and
  its `_out.txt`, and the orchestrator lists the folder before trusting any
  verdict. (k3_*, k3r_*)
- **Full-cache 52-week-low counts are contaminated by leveraged and inverse
  ETFs.** Inverse products sit at 52w lows exactly when the index is near its
  high, so a "new lows under a near-high index" count on the whole cache
  manufactures the state. Build breadth on stocks only, and report the
  effective N of today's reading (today: 11 of 161 tape stocks, 54 of 835
  cached stocks, and one ETF, XLU). (k2_c3b_short.py)
- **Grep the break side before calling a ratio cell unswept.** The surface map
  called the downside GDX/GLD break new because the registry's headline ratio
  cells are upside thrusts; the break side was closed on 2026-09-02
  (a2_c4_gdx_gld_pair.py). Search both signs of any spread.
- **A September quarter-end window contains the SEP FOMC.** QE-9 to QE in
  September starts inside the post-decision window, so any September
  quarter-end cell must be split from the post-FOMC drift before it is read.
  The dollar's September 14-4 since 2008 was exactly that drift. Quarter-end
  cells also owe the ordinary month-end control, which killed all three
  classes this morning. (k1_c7_fomc_split.py)

### Calendar finding, filed because it opens and closes a new anchor

- **The Rosh Hashanah to Yom Kippur window, first measurement in this repo.**
  Dates: Gauss Passover + 163 days for 1 Tishrei, Yom Kippur = +9, validated
  against 11 known dates and the molad arithmetic with zero mismatches
  1995-2030 (anchors in k2_c1_anchors.csv, reusable). The published volume
  regularity is REAL (SPY volume median 0.805x its +/-10 session neighbours on
  Yom Kippur, 0.830x on Rosh Hashanah day 1, against 0.998 for all sessions),
  and the price effect is not: short SPY YK-2 to YK pays -0.059% on 9-17,
  placebo rank 9 of 13, 2018+ 1-7. The full-window SPY short's +0.480% is
  2008's +18.57%. Byproduct rows NOT examined further: short IWM over the full
  window +1.177% on 17-9 (sign p 0.084), long SPY YK close +10 +0.631% on 17-9
  against +0.345% for all Sep/Oct days (median below the control's). Neither
  was slot-controlled. (k2_c1_yom_kippur.py)

### Cells swept and empty

- **Short SPY from two sessions before Yom Kippur to the Yom Kippur close.**
  Above; -0.073pp against the same calendar slot, -0.419pp against tdom-matched
  Sep/Oct, midterm 1-5, quad witching inside the rung 0-2. (k2_c1_yom_kippur.py)
- **Quarter-end window dressing on the nine SPDRs (top-2 minus bottom-2 by
  63d, QE-9 to QE).** +0.056% over 105 quarters against +0.101% at ordinary
  month-ends (Welch t -0.13), 0.7x cost; September 13-5 before 2018 and 3-5
  after; the 162-stock decile form 2-6 in 2018+ Septembers; run-in and
  post-quarter reversal uncorrelated (+0.038). (k1_c2_window_dressing.py,
  k1_c2_stocks.py)
- **Long EWJ into the March and September Japanese book closes.** Mar+Sep
  +0.607% at 27-26 against Jun+Dec +0.890% at 32-21; QE-2 to QE gives back
  -0.410% (20-33) where the dividend-reinvestment story places the buying;
  September 2018+ 2-6; JPY=X-adjusted September 13-13. (k1_c4_ewj_qe.py,
  k1_c4_sessions.py)
- **Long the dollar from QE-9 into the quarter-end close.** +0.046% over 106
  quarters against -0.038% at month-ends (t +0.51); December, the funding
  peak, 8-18; the last four sessions carry nothing since 2008; post-QE the
  dollar keeps rising (+0.228%, 62-44). (k1_c7_dollar_qe.py, k1_c7_fomc_split.py)
- **New-low breadth under a near-high index.** Tape long +0.083% at h=5
  against +0.191% for near-high days without the extreme; the short pays in
  31% of 54 tape neighbours against 93% of ETF neighbours; stocks-only
  [90,95) percentile bucket -0.275% to the short on 47-62.
  (k2_c3_newlow_breadth.py, k2_c3b_short.py)
- **Long GDX against beta-GLD at a downside ratio low.** At the 252 low -0.045%
  at h=5 over 23 (2018+ -1.173%) while a GDX flush without the ratio pays
  +0.804%. Second closure of the break side. (k3r_c5_parent.py, k3_c5_r1.py)
- **Short a refiner at a 252 high against XLE on a crude down day.** 0-3 at
  h=10 (-3.587%); 0 of 18 neighbours positive at h=5 or h=10; VLO outruns USO
  +4.72% over the next five sessions. Refiners at highs keep beating XLE and
  crude whatever crude does that day. The long side was NOT checked for cost,
  era or placebo. (k3r_c6_r1.py, k3r_c6b_neighbours.py, k3_c6_r1.py)
- **Long IEF against 0.523 TLT after a belly-led five-day selloff.** Negative
  at every horizon (h=5 -18.6 bp on 2-5); the TLT leg keeps parent dates that
  paid -0.227% and drops ones that paid +0.053%; the IEF 5d-floor parent is
  0.82x cost. A belly-led five-day selloff keeps going: the belly does not
  catch back up to the long end. (k3r_c8_r1.py, k3_c8_r1.py, k3_c8_neigh10.py)
"""


def main() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    if "## 2026-09-17:" not in text:
        REGISTRY.write_text(text.rstrip("\n") + SECTION, encoding="utf-8")
        print("registry: appended 2026-09-17 section")
    else:
        print("registry: section already present")

    w = load_watchlist()
    entries = w["entries"]
    for e in entries:
        if e["title"].startswith("Short regional banks against the big-bank index"):
            e["note"] = (str(e.get("note") or "") + " | 2026-09-17: breadth live (9 of 11 at "
                         "r5 <= 20) but the median 63d rank was 29.0, the broken form, not "
                         "the intact >= 70 cell.").strip(" |")
        if e["title"].startswith("Short a large bank against XLF"):
            e["note"] = (str(e.get("note") or "") + " | 2026-09-17: GS -1.43 ATR (below 1.5); "
                         "USB -2.35 and PNC -2.14 ATR intraday-led on a -0.44% SPY, both "
                         "outside the cell's eight-name universe.").strip(" |")

    titles = {e["title"] for e in entries}
    new = [
        {"added": TODAY,
         "title": "Long the dollar into the September quarter-end close",
         "cell": "quarter-end x dollar_fx, split from the SEP FOMC window",
         "trigger": (
             "A QUARTER-END PREMIUM THAT SURVIVES REMOVING THE FOMC WINDOW. September "
             "QE-9 to QE since 2008 goes 14-4 at +0.696% (sign p 0.015), and with the "
             "decision before the entry 6-0 at +1.365% (sign p 0.016), but the identical "
             "windows measured as post-decision drift are the same 14-4, and non-quarter "
             "FOMC months pay +0.374% on 47-28, so the quarter-end adds nothing. Pooled "
             "quarter-ends beat ordinary month-ends by only +0.084pp (t +0.51), +0.210pp "
             "since 2008. TURNS ON only if quarters with NO FOMC decision inside the "
             "QE-9..QE window beat ordinary month-ends by >= +0.25pp since 2008 (not yet "
             "measured). Standing blockers: December, the funding peak, is 8-18 (6-12 "
             "since 2008), and a DX 5d rank >= 95 at the signal is 2-5. A September form "
             "is yesterday's post-FOMC long-DX idea and must not ship under this label."),
         "script": "scratch/pitch_checks/2026-09-17/k1_c7_fomc_split.py",
         "source": "stand_down",
         "expires": "2026-10-08",
         "note": "2026-09-17: closest #1 in the stand-down."},
        {"added": TODAY,
         "title": "Short the quarter's two best SPDRs against its two worst from the quarter-end close",
         "cell": "quarter-end x sectors, the post-quarter reversal half of window dressing",
         "trigger": (
             "A DATE, then an era split. Winners minus losers (top-2 minus bottom-2 SPDRs "
             "by 63d) from the quarter-end close to QE+5 run -0.367% over 105 quarters "
             "(44-61) against -0.007% at ordinary month-ends, so the reversal pair earns "
             "about +0.37%. Unchecked beyond that one row, and a byproduct of the dead "
             "run-in cell (September 2018+ 3-5; run-in and reversal uncorrelated at "
             "+0.038), so it owes a charge for the walk. CHECK on the 2026-09-30 morning "
             "(anchor = the 09-30 close, h=5): TURNS ON if the 2018+ reversal clears "
             "+0.40% (5x an 8 bp two-leg round trip) and the September subset is not "
             "wrong-signed."),
         "script": "scratch/pitch_checks/2026-09-17/k1_c2_window_dressing.py",
         "source": "stand_down",
         "expires": "2026-10-01",
         "note": "2026-09-17: closest #2 in the stand-down; enterable only at a quarter-end close."},
    ]
    added = 0
    for n in new:
        if n["title"] not in titles:
            entries.append(n)
            added += 1
    save_watchlist(w)
    print(f"watchlist: {added} added, {len(entries)} active")


if __name__ == "__main__":
    main()
