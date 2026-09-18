"""Post-publish watchlist update, 2026-09-14 stand-down.

W18 fired its dose arm and died: its trigger is rewritten to the out-of-sample
arm the checker derived. W34 gets a note that its overlap-based arm is retired
by the 2026-09-08 owner decision. Two near-misses are parked.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa: E402

TODAY = "2026-09-14"
EXPIRES_15TD = "2026-10-05"

d = load_watchlist()
entries = d["entries"]


def find(prefix: str) -> dict:
    hits = [e for e in entries if e["title"].startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


w18 = find("The duration-neutral curve position")
w18["trigger"] = (
    "OUT OF SAMPLE ONLY. KILLED 2026-09-14 on the re-arm itself. The dose arm fired "
    "(252-session change +94.3 bp on 2026-09-11, ^TNX 4.975 at its 252 max) and the +88 bp "
    "dose did not filter: as its own cell it pays +34.5 bps at h=8 on 25 episodes (20-5) "
    "against +34.7 bps on 29 for the +78 bp parent, and absolute-change buckets are "
    "non-monotone (+45.6 / -2.3 / +20.1 / +59.2 bps at [88,100) / [100,113) / [113,150) / "
    ">=150, Spearman +0.021). The parent owes the charge for the walk that found it: "
    "P 0.8187 over 360 cells (3 vehicles x 10 horizons x 6 proximity rungs x thresholds "
    "{78, 88}), null-max median 64.7 bps. No live-state rung can cure a multiplicity charge. "
    "TURNS ON when at least 3 declustered +78 bp episodes (^TNX within 0.25% of its 252 max, "
    "252-session change >= +78 bp, filter-then-decluster at gap 10) signalled AFTER 2026-09-01 "
    "have realized a mean h=8 curve return of at least +22.1 bps (5x the 4.423 bp round trip "
    "including borrow). OOS episode 1 is the 2026-09-09 signal (entry 09-10 MOC, exit 09-22 "
    "close); 09-10 and 09-11 belong to the same episode. Do not re-arm on another dose, "
    "threshold or era split: before 2018 the gate trails its own complement (+10.4 against "
    "+16.7 bps) and it only filters from 2018 (+51.8 against -9.6). On file and not load-"
    "bearing: FOMC inside the hold 6-0 at +41.2 bps; the thin-start-then-dose path 5-0 at "
    "sign p 0.031 but +9.9 bps (2.2x cost) after dropping the best episode."
)
w18["script"] = "scratch/pitch_checks/2026-09-14/k1_c1_dose_b.py"
w18["note"] = ((w18.get("note") or "") + " | 2026-09-14: dose arm FIRED (+94.3 bp) and the "
               "cell was killed on firing; trigger rewritten to an out-of-sample realized-return "
               "arm (k1_c1_dose.py, k1_c1_dose_b.py, k1_c1_parent_c.py).").strip(" |")

w34 = find("The pooled sector triple rank floor")
w34["note"] = ((w34.get("note") or "") + " | 2026-09-14: the arm text ('a reason to exist "
               "beside the book', <20 book signals in the window) is a portfolio-overlap "
               "criterion that the 2026-09-08 owner decision retires for the Daily Pitch. Restate "
               "it as a live-state arm (a nine-SPDR member at the 5/21/63 floor) before scoring. "
               "Carry the 2026-09-14 finding into any re-score: the 22-ETF washout family pools "
               "to +0.12pp at h=10 (t 0.38), and its strong midterm slice pays only with SPY "
               "below its 200-day (+1.68pp at h=5 against -0.26pp above). No nine-SPDR member "
               "held the floor on 2026-09-11 (nearest XLI r5 20.6 / r21 3.6 / r63 5.2)."
               ).strip(" |")

entries.append({
    "added": TODAY,
    "title": "Long XLV against 0.71 SPY after a healthcare complex flush, as a family effect",
    "cell": "sectors, flush x beta-neutral residual against the index",
    "trigger": (
        "THE FAMILY EFFECT CLEARING COST, or XLV separating from its family. The cell (XLV "
        "5-day rank <= 1 with >= 3 of XLV/IBB/XBI/IHI at r5 <= 5), long XLV against 0.71 SPY "
        "(full-sample beta), pays +0.702pp of excess at h=10 over 21 episodes, 16-5, sign p "
        "0.0133, matched on SPY's forward-return bucket +0.86pp at 15-6, both eras positive "
        "(+0.81 pre-2018, +0.70 after), top-two episodes 54%. It is flat at h=1..4 (<= +0.08%). "
        "The same rule on 8 sector complexes is homogeneous (Cochran Q 6.75 on 7 df, I-squared "
        "0%) with a fixed-effect common excess of +0.294pp, 4.5x a 6.5 bp pair round trip, and "
        "XLV ranks 3rd of 8 (max-of-8 P 0.904 beta-neutral, 0.776 equal-dollar). TURNS ON when "
        "the 8-complex common excess reaches >= +0.33pp at h=10 (5x cost; +0.294pp today), OR "
        "XLV's max-of-8 P falls to <= 0.10, OR the reference class is widened with the missing "
        "complexes (KBE, KIE, IGV, SOXX, IDU, PBJ are not in master_prices) and the common "
        "effect clears 5x on the wider class. THREE DEBTS, none optional. (1) The residual is "
        "midterm-concentrated: +1.441pp at 9-1 against +0.018pp at 7-4, a split nobody "
        "pre-registered. (2) Hedge ratio: XLV's trailing 63-day beta to SPY was -0.23 on "
        "2026-09-11 and its 252-day beta 0.29; at the 63-day beta the residual falls to "
        "+0.38pp. (3) The live definition is the best of 18 neighbours; r5 <= 2 halves it to "
        "about +0.53pp. Do not pitch the outright (h=10 excess -0.03pp against XLV's drift) or "
        "the equal-dollar pair (its edge is the short SPY leg)."
    ),
    "script": "scratch/pitch_checks/2026-09-14/k2_c2_xlv_flush_b.py",
    "source": "stand_down",
    "expires": EXPIRES_15TD,
    "note": "2026-09-14: near-miss in the stand-down; family table in k2_c2_xlv_flush_d.py.",
})

entries.append({
    "added": TODAY,
    "title": "Beta-hedged short SVXY the session after a 10% one-day VIX crush, 2018+",
    "cell": "volatility price-state, the ungated parent of the pre-FOMC crush candidate",
    "trigger": (
        "COST, and the dose has to turn the right way. Short SVXY hedged against SPY at the "
        "era beta (1.48-1.51), entered MOC the session after any >= 10% one-day ^VIX fall in "
        "the -0.5x era (2018-03+), pays +0.335% at h=1 over 101 crush days, 62-39, sign p "
        "0.014, t 2.14, 2.8x a ~12 bp two-leg round trip. TURNS ON at 5x cost, i.e. +0.60% at "
        "h=1 on the hedged form, AND a monotone dose: the >= 12% crush bucket must pay at least "
        "the >= 10% cell, against +0.097% at h=1 and -0.099% at h=2 today. Two things already "
        "settled, do not re-run them: the FOMC-ahead gate SUBTRACTS (+0.123% with an FOMC in "
        "the next three sessions against +0.361% without), and the pre-FOMC crush does not get "
        "re-bid (^VIX to the decision close 1-7 at -3.86%, long UVXY 2018+ 0-4). Adjacent to "
        "the registry's dead 'SVXY as a pre-FOMC leg' and to W33's SPY-residual rule; the hedged "
        "construction is what makes this a different object."
    ),
    "script": "scratch/pitch_checks/2026-09-14/k3_c6_vix_crush_prefomc_b.py",
    "source": "stand_down",
    "expires": EXPIRES_15TD,
    "note": "2026-09-14: near-miss in the stand-down (closest #2).",
})

d["generated"] = TODAY
d["asof"] = TODAY
save_watchlist(d)
print(f"watchlist saved: {len(entries)} active entries, {len(d.get('expired', []))} retired")
