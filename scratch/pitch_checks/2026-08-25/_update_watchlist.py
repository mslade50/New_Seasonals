"""Post-publish watchlist maintenance for 2026-08-25.

Appends today's three near-misses with the number each turns on at, stamps the
morning's verdict onto every entry that was checked in the surface map, and
prunes anything expired. Nothing here is a trade; it is the record stage B1
owes a verdict against tomorrow.
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WL = ROOT / "data/pitch_watchlist.json"

TODAY = "2026-08-25"
data = json.loads(WL.read_text(encoding="utf-8"))
entries = data["entries"] if isinstance(data, dict) else data
print("before:", len(entries))

# --- verdict notes for entries the surface map checked this morning ---------
NOTES = {
    "Long TLT from the NFP close to +3td, with the long end at its 52w floor":
        "2026-08-25 verdict: PASS. Still midterm; NFP is +8 td, at the horizon cap. Arms 2027-01.",
    "Credit-quality divergence, long LQD against short HYG at joint 52w extremes":
        "2026-08-25 verdict: PASS. State still live (HYG -0.11% off its high, LQD +0.72% above its low) "
        "and the trigger is episode count, still 4 declustered since 2007 against the 8 required.",
    "Long SVXY overnight into the CPI print, MOC the eve to MOO on the print":
        "2026-08-25 verdict: PASS. Next CPI is 2026-09-11, +12 td, outside the 10 td horizon cap.",
    "Long GLD on a miner-led thrust the metal has not joined":
        "2026-08-25 verdict: PASS on the fourth condition. GDX r5 rank 95.6 and GLD r5 92.5 both fire, "
        "but GLD is -13.96% off its 52w high against the >-10% trend rung. Independent confirmation from "
        "today's C7 check: the LONG side of this thrust state pays +0.844% (edge +0.575pp, 58.5% hit) "
        "while the fade is wrong-signed at all ten horizons, so the sign of the parent is not in doubt "
        "- only the drawdown conditioner is. (c2_c7_gdx_thrust_fade_r1.py)",
    "Long XLE on a crude one-day thrust in the 5 to 6 percent band":
        "2026-08-25 verdict: PASS. USO's 5d move is +1.47%, nowhere near the [5%,6%) one-day band.",
    "Long TLT with the whole investment-grade complex pinned at 52w lows":
        "2026-08-25 verdict: PASS twice. The tight rung is NOT live (TLT +1.49% above its 52w low against "
        "the <=0.5% rung; IEF +0.90%, LQD +0.72% do clear theirs), and the freshness leg fails anyway "
        "- the last tight day was 2026-08-18, 4 sessions back against the >=10 requirement.",
    "Long SPY on a skew spike alone, no volatility condition":
        "2026-08-25 verdict: PASS. SKEW's 5d rank is 67.5 against the >=95 leg, and the midterm block stands.",
    "Fade a crude thrust out of a deep base with a macro print inside the hold":
        "2026-08-25 verdict: PASS. USO r5 rank 55.2 (needs >=90) and r63 rank 27.0 (needs <=20).",
    "Long the medical-device thrust, IHI at a 21d rank of 100 out of a drawdown":
        "2026-08-25 verdict: PASS. IHI's 21d rank is 98.0, not 100, and the family-wise p 0.933 "
        "reference-class blocker is unchanged.",
    "Long China's five-day break inside an intact thrust, FXI while EEM holds":
        "2026-08-25 verdict: PASS, and a premise correction is owed to anyone re-reading it. FXI's 5d rank "
        "is 65.9 against the <=20 trigger. Separately, this morning's recon claimed FXI-EEM 63d was "
        "+4.24pp at PIT 99.6; on valid sessions it is -0.03pp at PIT 90.5 - the two are LEVEL over 63 "
        "sessions, not diverging. The error was a pad-filled union-calendar panel. (c3b_c8_premise_forensic.py)",
    "Long TLT on the NOVEMBER month-position effect":
        "2026-08-25 verdict: PASS. Parks to a date, trading days 4-12 of November 2026.",
    "Short SPY with the index at a 52w high while the long end sits at a 52w low":
        "2026-08-25 verdict: PASS. SPY is 1.85% below its 52w high against the <=0.5% rung.",
    "Long TLT into the month-end close, ungated, entered nine sessions before it":
        "2026-08-25 verdict: PASS on its own second condition. Today IS ME-4, inside the flat ME-3..ME-7 "
        "entry band, but the entry's stated blocker is literally 'NOT August' and August ME-5 is 5-of-11 "
        "at -0.510% since 2015. Filed alongside it today: the FX translation of this same month-end flow "
        "was tested and CLOSED (the fix session pays -0.55 bp at a 45.6% hit and flips positive post-2020), "
        "so the month-end anchor is now swept on equities, rates and FX. (d1_c5_monthend_fx_r1.py)",
    "Long SPY on a volatility pop inside an already-calm tape":
        "2026-08-25 verdict: PASS, and narrowly. VIX's 21d rank of 19.0 clears the calm-tape leg, but the "
        "day's VIX move was +4.76% against the >=+5.0% pop rung - short by 0.24pp.",
    "Long gold on an unconfirmed rate rise, with both dials at force":
        "2026-08-25 verdict: PASS. The dollar leg clears easily (DX 21d rank 0.8, PIT 0.4 on the 21d return "
        "- the most oversold dollar of the past year); the yield leg does not, at roughly +0.05pt over 21 "
        "sessions against the +0.20pt floor.",
    "Long technology against healthcare after a rotation gap, in calm near-high tape":
        "2026-08-25 verdict: PASS on its own trigger (one-day XLV-XLK gap +1.18pp against >=+3.0pp), but "
        "READ THIS BEFORE RE-TESTING. The FIVE-day version of this rotation hit +9.98pp today, the 99.6th "
        "full-sample percentile, and was pitched and killed as C1. Two findings transfer directly. (1) The "
        "5d form is largely THIS form in disguise: 95.4% of rung>=8 days and 100% of rung>=9/10 days "
        "contain a >=3pp single-day gap, and 2026-08-18's +4.07pp print - the exact session this entry was "
        "built on - plus 08-19 are 86.6% of today's spread. Episode containment, not day-level overlap, is "
        "the right test. (2) This entry's closing line that the naked long beats the pair is now measured "
        "harder: count-matched, XLK 5d <= -9.82% ALONE pays +1.069% against the spread trigger's +0.555%, "
        "so the defensive leg is decoration. Neither changes this entry's own turn-on. (a1b_c1_round2.py)",
    "Short the dollar on a rate rise the currency does not confirm":
        "2026-08-25 verdict: PASS. ^TNX's 21d RETURN rank is 45.6 against the >=65 leg. The dollar half is "
        "at its most stretched in a year (DXY 21d PIT 0.4) and the rate half is absent, which is the same "
        "one-sided state as 2026-08-24.",
    "Long crude through Jackson Hole, entered six sessions before the conference":
        "2026-08-25 verdict: PASS. The JH-6 anchor was 2026-08-20, five sessions gone (today is JH-3), and "
        "XLE is -1.00% off its 52w high, which the entry's own second condition forbids.",
    "Short TLT after a big up day from inside the 52-week low zone":
        "2026-08-25 verdict: PASS. TLT's one-day move was -0.35% against the >=+1.5% thrust rung.",
    "Short regional banks against the big-bank index on a breadth washout":
        "2026-08-25 verdict: PASS. KRE's 5d rank is 7.9 so the breadth state is live, but the trigger is a "
        "cost threshold on ex-crisis history that a new episode cannot move.",
    "Long high yield across the Jackson Hole speech, entered five sessions before it":
        "2026-08-25 verdict: PASS. The JH-5 anchor was 2026-08-21 and is gone; the anchor is closed on credit.",
    "The duration-neutral curve position, long IEF against short 0.52 TLT, with the 10-year yield at a "
    "52-week high":
        "2026-08-25 verdict: PASS, and it is the closest this has been - ^TNX is at 99.72% of its "
        "trailing-252 high, i.e. within 0.28% against the 0.25% rung. The COST turn-on is unmoved (needs "
        "30 bps at h=8, ladder tops at 22.2). Independently, the entry's second blocker forbids entry this "
        "week regardless: JH-spanning episodes are 0-for-6, and any hold entered today spans Friday's speech.",
    "The NARROW energy thrust cluster, two or three names at z10 above 2 rather than five":
        "2026-08-25 verdict: PASS. Its three prereg debts (own pre-registration, narrow-form reference "
        "class, August weakness) are all unpaid. Related evidence filed today from the C4 energy check: "
        "the four-name complex {XLE, XOP, OIH, USO} has PC1 83.0% and 1.42 effective names of 4, which "
        "reinforces this entry's own effective-N caveat on the 11-name version. (c1_c4_oih_xop_r1.py)",
    "Cross-sectional new-high breadth on a survivorship-free universe, with the index further off its high":
        "2026-08-25 verdict: PASS on both legs, and both moved the WRONG way. SPY is -1.85% off its high "
        "against the >2.0% requirement, and raw-21d fragility is 64.5 against the <=50 requirement. The "
        "dial half is now emphatically out of range: today's ma10(63d) is 89.5 against a trigger-population "
        "median of 24.8 and an all-time max of 80.6.",
}

for e in entries:
    note = NOTES.get(e.get("title", ""))
    if note:
        e["note"] = note

# --- today's near-misses ----------------------------------------------------
NEW = [
    {
        "added": TODAY,
        "title": "Long OIH outright, no short leg, at a 63-day services-versus-E&P extreme",
        "cell": "energy, intra-sector dispersion",
        "trigger": "THE RECORD, and it is 4 wins away. Long OIH MOC (no short leg) when the OIH-minus-XOP "
                   "63-day return spread sits at or below its 2.5th PIT trailing-252 percentile, h=10: "
                   "+0.934%, edge +0.706pp over OIH's own drift, 28-23, and 15.6x a 6 bp one-leg round "
                   "trip. It is the residue of a PAIR that was pitched and killed the same morning - the "
                   "pair is wrong-signed at h=1/2/3/5 (-0.209% at h=5, 35-44, sign p 0.870, -1.7x cost) "
                   "because the short XOP leg contributes -0.439pp while the long leg contributes +0.763pp. "
                   "TURNS ON at 32 of 51 wins, which is sign p 0.046; it stands at 28, so four more winning "
                   "episodes without a loser, or six of the next seven. TWO standing caveats no threshold "
                   "cures. (1) The one positive horizon rests on one episode: drop-best leaves -0.018% and "
                   "drop-best-2 leaves -0.200%, so the record has to thicken rather than the mean rise. "
                   "(2) The complex is one factor - PC1 83.0%, 1.42 effective names of 4 - so this is "
                   "levered crude beta wearing a services label, and the book has historically been SHORT "
                   "this state (13 energy-family ledger signals in these windows, all Overbot Vol Spike "
                   "shorts at avgR +1.083). Today's live reading: -16.78pp, PIT 1.19, so the state IS "
                   "firing; it is the record that is short.",
        "script": "scratch/pitch_checks/2026-08-25/c1_c4_oih_xop_r1.py",
        "source": "near_miss",
        "expires": "2027-02-25",
    },
    {
        "added": TODAY,
        "title": "The sector washout into a 52-week high at h=7, as a FAMILY effect rather than an XLI call",
        "cell": "sectors, washout x trend-intact, cross-sector",
        "trigger": "HETEROGENEITY, and the honest form of this is a family trade or nothing. A sector "
                   "washing out (5-day rank <= 5) while within 5% of its 52-week high pays a POOLED "
                   "+0.900% at h=7 across the nine SPDRs, and XLI's own cell is +1.613% with drop-best-3 "
                   "still +0.906% at 32.3x cost - a genuine shelf spanning h=5..h=9 (+0.677 / +0.853 / "
                   "+1.613 / +1.784 / +1.802). What kills the XLI VERSION is that nothing distinguishes "
                   "XLI: Cochran Q is 4.70 on 8 df at p=0.789 (homogeneous, I-squared 0), XLI ranks 2 of 9 "
                   "by |t| BEHIND XLP, and the permutation max-of-9 null gives P(max |t| >= 2.65) = 0.268. "
                   "TURNS ON when Cochran Q p < 0.10 with XLI first of 9 by |t|, which at today's episode "
                   "count needs roughly four more non-2024 XLI episodes averaging >= +2.0% at h=7 (ex-2024 "
                   "the cell is 4-3 at +1.174%, and 6 of 13 episodes are 2024 carrying 61% of the return). "
                   "THREE caveats that survive any threshold. (1) The 'intact trend' clause the idea was "
                   "named for is a NEGATIVE-value filter: bare washout +0.368% at h=3, joint cell +0.234%, "
                   "broken-trend complement +0.418%. Whatever this keys on, it is not an intact 63-day "
                   "trend. (2) The near-high gate that does the real work is a bull-tape selector - 100.0% "
                   "of trigger days sit above SPY's 200d against a 71.6% base, with ZERO observations "
                   "below. (3) The literal pitched rung (r5<=2.0, r63 in [40,50], within 5% of high) has "
                   "three days in history, so anything measured is already a loosened form. Dial max on "
                   "any episode is 68.6 against today's 89.5, and 8 of 13 episodes predate the dial "
                   "entirely. If this ever arms, it arms POOLED across the family, not on one sector.",
        "script": "scratch/pitch_checks/2026-08-25/a3c_c9_h7_refclass.py",
        "source": "near_miss",
        "expires": "2027-02-25",
    },
    {
        "added": TODAY,
        "title": "The utilities washout with the long end hit ALONGSIDE it, which is the negation of the "
                 "cell pitched on 2026-08-25",
        "cell": "utilities x rates, washout co-movement",
        "trigger": "THE RATES LEG FLIPPING SIDES, and read the debt before touching this. The morning "
                   "pitched 'XLU washed out while TLT sat mid-range' on the theory that an equity rotator "
                   "is a better seller than a rates repricing; the tape says the opposite. XLU 21-day rank "
                   "<= 5 with TLT ALSO hit pays +0.858% at h=5 over 28 episodes at a 75.0% hit (sign p "
                   "0.006) and +0.971% at h=3 on a 64.3% hit, while the pitched mid-range version pays "
                   "+0.090% against XLU's all-days base of +0.132%, i.e. -0.042pp against doing nothing. "
                   "TURNS ON at XLU 21d rank <= 5 AND TLT 21d rank < 25 on the same session; today TLT's "
                   "21d rank is 48.4, so the state is not live. THREE DEBTS, none optional, because this "
                   "was recovered from the corpse of a kill and the anti-rescue rule applies. (1) It owes "
                   "its own pre-registration and a FORWARD re-derivation - it is a post-hoc sign flip found "
                   "inside a kill report, which is exactly the shape the 2026-08-24 narrow-energy entry was "
                   "made to pay for. (2) The nine-sector reference class has NOT been run on this form; on "
                   "the parent it put XLU 8 of 9 with Cochran Q p=0.946, so it must be shown this is not an "
                   "any-sector effect. (3) The parent cell is itself a strict subset of the dead 2026-08-12 "
                   "rank21<=5 utilities cell, and utilities are now dead in eight expressions, so the prior "
                   "here is bad and the burden is correspondingly high.",
        "script": "scratch/pitch_checks/2026-08-25/b1_c3_xlu_washout_tlt_fine.py",
        "source": "stand_down",
        "expires": "2027-02-25",
    },
]

have = {e.get("title") for e in entries}
for n in NEW:
    if n["title"] in have:
        print("  already present, skipping:", n["title"][:60])
    else:
        entries.append(n)
        print("  added:", n["title"][:70])

# --- prune expired ----------------------------------------------------------
kept, dropped = [], []
for e in entries:
    exp = str(e.get("expires", "")).strip()
    (dropped if exp and exp < TODAY else kept).append(e)
for d in dropped:
    print("  pruned (expired", d.get("expires"), "):", d.get("title", "")[:60])

if isinstance(data, dict):
    data["entries"] = kept
    data["generated"] = TODAY
    out = data
else:
    out = kept

WL.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("after:", len(kept), "->", WL)
