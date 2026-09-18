"""Post-publish housekeeping for 2026-09-16: registry append + watchlist update."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

REGISTRY = ROOT / "data" / "pitch_negative_registry.md"
TODAY = "2026-09-16"

SECTION = """

## 2026-09-16: the post-decision anchor, swept across six classes on an FOMC day

Ten candidates over four novelty axes and six asset classes, four adversarial
checkers plus a red team, one survivor shipped (long DX futures from the
decision close, h=5, grade B). The 09-15 tape: ^TNX closed exactly at its
trailing-252 max (4.996), TLT/IEF/LQD at 252 lows, USO +27.85% in 21d, XLU and
XLI at a simultaneous 5/21/63-day rank floor, SPY 6.15% above its 200d, on an
FOMC decision day that was also a VIX expiry.

### Method traps

- **The post-FOMC yield relief is an ANNOUNCEMENT-SESSION object, not a
  post-decision one, and with the ten-year at its high it does not arrive at
  all after the close.** On 31 decisions with the eve ^TNX within 15 bp of its
  252 max, ^TNX rises +4.2 bp by h=3 and TLT loses 0.48%, against -0.3 bp and
  +0.103% on the other 173. The Hillenbrand-style relief sits on eve close to
  decision close (TLT hike-regime +0.354%, 39-16, sign p 0.001; at the 252 max
  +0.663%, 6-1). Any rates, gold, utilities or credit idea entered MOC on the
  decision close with the ten-year at a high starts from this wrong-signed
  base. (kA_c1_c5_c6_postfomc_b.py, kB_c2_xlu_fomc.py)
- **Quad witching inside an FOMC hold is a proxy for the QUARTERLY meetings,
  and before 2011 those carried no projections.** The apparent weakness of
  "quad in hold" on the long-dollar cell (h=5 +0.090%) is pre-2011 Mar/Jun/Sep/Dec
  meetings (-0.200%, N 43, Welch t -2.71 vs other months); SEP meetings since
  2011 pay +0.358% (42-19). Split by SEP vs not before reading an expiry
  collision on any FOMC cell; the equity expiry has no channel into FX.
  (kR_c10_redteam.py)
- **A decision-close anchor can filter a flush the wrong way.** HYG z10 <= -2
  (tape convention) on any day pays +0.549% at h=3 over 47 (34-13); the 5 that
  land on a decision close pay -0.129% (filtering -0.408pp). Event anchors on
  price-state cells owe filter_vs_reanchor before any event story.
  (kB_c4_hyg_fomc_flush.py)

### Cells swept and empty

- **Long TLT from the FOMC decision close with the ten-year at a 252 high.**
  h=3 -1.53% (2-5, tdom-matched -1.64pp, 2018+ 0-5), within-2% -0.96% (5-11),
  placebo 10 of 11; the gate selects yield continuation (hike-regime decisions
  off the max +0.17%). (kA_c1_c5_c6_postfomc.py, _b)
- **Short the dollar from the decision close.** Wrong-signed parent: h=3
  -0.172% over 204 (t -2.73, tdom -0.21pp, placebo 11 of 11). Its sign flip
  shipped the same morning. (kA_c1_c5_c6_postfomc.py)
- **Long gold from the decision close with the ten-year at a 252 high.** One
  episode (2006 +2.87%, ex -0.04%); parent h=3 tdom -0.54pp over 173, worst
  anchor on the ladder; GC=F flips by era. With the 09-14 and 09-15 kills, gold
  around a yield-thrust FOMC is closed on all three rungs. (kA_c1_c5_c6_postfomc_b.py)
- **Long SPY from the decision close to the end of FOMC week zero.** Placebo
  11 of 11 at h=1/2/3 over 204; the FOMC-cycle premium is the decision session
  itself (+0.235%, t 2.75), before a MOC entry; midterm h=3 -0.470% on 23-30.
  (kB_c3_spy_postfomc.py)
- **Long XLU (and the seven-name rate-sensitive family) from the decision
  close with the ten-year at its high.** XLU h=5 -0.558% (tdom -0.68pp, placebo
  11 of 11); XLRE, IYR, VNQ, XHB, ITB, XLP and SPY all negative at h=2/3/5.
  Utilities now dead in nine expressions. (kB_c2_xlu_fomc.py, _b)
- **Long HYG from the decision close after a flush.** Above; the live
  duration-driven form matched once (2022-06-15, -0.67% at h=5 while ^TNX fell
  32.7 bp). (kB_c4_hyg_fomc_flush.py)
- **The nine-SPDR 5/21/63 triple floor (watchlist 35, restated as a live-state
  arm after the portfolio criterion was retired).** Real across history
  (+1.131% h=10 over 493, FE t 3.17, I-squared 0%) and dead in the live regime:
  midterm with SPY above its 200d -0.393pp over 28 clusters, never above
  +0.335pp across 18 neighbours; the 5d leg re-anchors (-0.792pp filtering);
  XLU 9 of 9. Confirms the 09-14 above/below-200d split on a second family.
  (kC_c7_triple_floor.py, _b)
- **Floored SPDRs against XLE at a 252 high.** The generic reversal pair again
  (paired diff vs long-two-worst/short-two-best -0.102%); 4 exact episodes;
  live regime -0.802% above the 200d. (kC_c8_floor_vs_high_pair.py)
- **Long EEM at a 63-day floor inside a year up more than 20%.** Above SPY's
  200d -0.404pp over 10 (4-6); 7 of 12 country ETFs; year > 30% flips to
  -2.391pp; not the 09-15 family state (that needs a year up >= 40%).
  (kC_c9_eem_floor.py, _b)
"""


def main() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    if "## 2026-09-16:" not in text:
        REGISTRY.write_text(text.rstrip("\n") + SECTION, encoding="utf-8")
        print("registry: appended 2026-09-16 section")
    else:
        print("registry: section already present")

    w = load_watchlist()
    entries = w["entries"]
    for e in entries:
        if e["title"].startswith("The pooled sector triple rank floor"):
            e["trigger"] = (
                "SPY BELOW ITS 200-DAY, restated 2026-09-16 as a live-state arm (the "
                "portfolio-overlap arm is retired). Scored in full on 2026-09-16 with XLU "
                "and XLI both at the 5/21/63 <= 10 floor: the pooled nine-SPDR form "
                "reproduces (+1.131% at h=10 over 493 episodes, excess +0.733pp, "
                "fixed-effect common +0.801pp at t 3.17, I-squared 0%) and every state "
                "except the live one pays +1.384pp, but midterm with SPY above its 200d "
                "pays -0.393pp over 28 date clusters (raw +0.015%, under cost) and never "
                "tops +0.335pp across 18 definition neighbours. TURNS ON at a SPY close "
                "below its 200d SMA while a nine-SPDR member holds the floor (midterm "
                "below the 200d pays +2.069pp at 15-4); on 2026-09-15 SPY sat 6.15% above "
                "it. Standing debts: the 5d leg re-anchors rather than filters "
                "(filtering -0.792pp against the 21/63 double floor), and XLU ranks 9 of "
                "9 at -0.590pp, so a utilities-heavy basket is the weakest form.")
            e["script"] = "scratch/pitch_checks/2026-09-16/kC_c7_triple_floor_b.py"
            e["note"] = (str(e.get("note") or "") + " | 2026-09-16: CHECKED and killed on the "
                         "live regime (closest #1 in the short slate); arm restated to SPY "
                         "below its 200d.").strip(" |")

    titles = {e["title"] for e in entries}
    new = [
        {"added": TODAY,
         "title": "Long TLT across the FOMC announcement session with the ten-year at its 252 high",
         "cell": "fomc_decision x rates, the eve-close-to-decision-close rung",
         "trigger": (
             "THE NEXT FOMC EVE, plus a reconciliation debt. Long TLT MOC on the eve "
             "close, exit at the decision close: hike-regime decisions pay +0.354% at "
             "39-16 (sign p 0.001) and decisions with ^TNX at its trailing-252 max "
             "+0.663% at 6-1 (sign p 0.062), within 2% +0.336% at 10-6. Found as a "
             "byproduct of the 2026-09-16 post-decision kill, so it owes a charge for "
             "that walk. TURNS ON at the 2026-10-27 eve close (FOMC 10-28) if ^TNX "
             "closes within 2% of its 252 max, AND only after it is reconciled against "
             "two registry kills of duration into the FOMC (09-01 gate attribution on "
             "the flattener, +0.017pp over 192 anchors; 09-15 80/80 bond-vol rung "
             "-0.306% at h=1, placebo 9 of 11). Those measured different vehicles and "
             "holds; say which object this is before it ships. Placebo ladder on the "
             "one-session hold is owed."),
         "script": "scratch/pitch_checks/2026-09-16/kA_c1_c5_c6_postfomc_b.py",
         "source": "short_slate",
         "expires": "2026-10-28",
         "note": "2026-09-16: closest #2 in the short slate; window was the 09-15 close, gone at publish."},
        {"added": TODAY,
         "title": "Long HYG after a SPREAD-driven five-day flush, the any-day parent",
         "cell": "credit price-state, flush split by the duration leg",
         "trigger": (
             "IEF NOT FLUSHED ALONGSIDE. HYG tape-convention z10 <= -2 on any day pays "
             "+0.549% at h=3 over 47 episodes (34-13) and +0.784% at h=5. The live "
             "2026-09-15 flush (HYG z10 -2.27) was duration-driven with IEF's 5d rank "
             "at 1.2, and that form pays +0.152% at h=3 and -0.023% at h=5 over 34 "
             "episodes (the registry's 09-14 dead cell). TURNS ON at a HYG z10 <= -2 "
             "close with IEF's 5d rank above 20; round 2 (declustering, era, "
             "concentration) has NOT been run on the spread-driven form and is owed "
             "before it is scored. The z10 convention must be stated: pitch_lab.zscore "
             "gives 10 episodes where the tape convention gives 5 on decision closes."),
         "script": "scratch/pitch_checks/2026-09-16/kB_c4_hyg_fomc_flush.py",
         "source": "short_slate",
         "expires": "2026-10-07",
         "note": "2026-09-16: closest #3 in the short slate."},
    ]
    added = 0
    for n in new:
        if n["title"] not in titles:
            entries.append(n)
            added += 1
    w["expired"] = []
    w["asof"] = TODAY
    save_watchlist(w)
    print(f"watchlist: {len(entries)} entries, {added} added, triple-floor arm restated")


if __name__ == "__main__":
    main()
