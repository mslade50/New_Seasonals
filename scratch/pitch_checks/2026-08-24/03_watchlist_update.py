"""Post-publish watchlist maintenance for 2026-08-24.

Three jobs, per SKILL.md "After publishing":
  1. append this morning's near-misses with the number that turns each on
  2. prune the entry the state builder listed as expired
  3. CORRECT W12, whose stated blocker was re-derived today and does not hold
     on the parent it parks (it was fitted on the oversold-GATED subsample)

Also refreshes the `note` on every active entry with today's verdict, so the
file carries the last time each was actually looked at rather than implying it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa

TODAY = "2026-08-24"

# verdict lines from 02_watchlist_verdicts.py, one per active entry, keyed by
# a distinctive fragment of the title
VERDICTS = {
    "NFP close": "2026-08-24 verdict: PASS. 2026 is midterm and the trigger is the cycle year; the next NFP (2026-09-04) is +9 td, at the horizon edge. Arms 2027-01.",
    "Credit-quality divergence": "2026-08-24 verdict: PASS. The state is still live (HYG -0.23% off its high, LQD +0.56% off its low) and the trigger is episode count, still 4 declustered since 2007.",
    "SVXY overnight": "2026-08-24 verdict: PASS. The next CPI is 2026-09-11, +13 td, outside the 10 td horizon cap.",
    "miner-led thrust": "2026-08-24 verdict: PASS on the fourth condition, which is the one added after 08-21 fired the first three. GDX r5 rank 98.4 and GLD r5 rank 94.0 both fire; GLD is -14.63% off its 52w high against the >-10% trend rung. Independently reconfirmed this morning: the deep-drawdown half pays -0.798% at h=5 (edge -1.012pp, 35% hit) on the C9 check, d3_c9_gold_yield_level.py.",
    "crude one-day thrust": "2026-08-24 verdict: PASS. USO's one-day move was +0.07%, nowhere near the [5%,6%) band.",
    "investment-grade complex": "2026-08-24 verdict: PASS, and the closest it has been — IEF (+0.70%) and LQD (+0.56%) both clear their rungs, TLT is +0.86% against the <=0.5% tight rung. AMENDED today: the IEF translation was tested and is dead (IEF at its own low predicts IEF -0.021% at a 49.2% hit; the join is 1.0x cost; TLT/IEF excess ratio 1.22 against a daily-sd ratio of 2.13). The shape belongs on TLT and only fresh — TLT on the drop-TLT rung, episode-first, ex-2022 is +0.289% at a 61.1% hit, t 2.19, 9.1x cost, N=18. Do not widen the rung to IEF/LQD to reach a sample. Freshness also fails today: the previous trigger day is 1 session back against the >=10 requirement. Evidence: a3_c10_ief_ig_rungs.py, a3b.",
    "skew spike": "2026-08-24 verdict: PASS twice. SKEW's 5d rank is 82.1 against the >=95 leg, and the midterm block stands regardless. Arms 2027-01.",
    "deep base": "2026-08-24 verdict: PASS. USO r5 rank 79.4 (needs >=90) and r63 rank 29.4 (needs <=20); neither leg fires.",
    "medical-device": "2026-08-24 verdict: PASS. IHI's 21d rank is 99.6, not 100, and the reference-class blocker (family-wise p 0.933) is unchanged.",
    "China's five-day break": "2026-08-24 verdict: PASS. FXI's 5d rank is 87.7 and the trigger needs <=20, so the state is the opposite of the cell.",
    "NOVEMBER month-position": "2026-08-24 verdict: PASS. Parks to a date, trading days 4-12 of November 2026 (~2026-11-05).",
    "52w high while the long end": "2026-08-24 verdict: PASS. SPY is 1.56% below its 52w high against the <=0.5% rung, so the joint state is not live.",
    "month-end close, ungated": (
        "2026-08-24 verdict: PASS on the calendar (August's ME-9 was 2026-08-18; today is ME-5), "
        "and the entry is CORRECTED AND SUPERSEDED — see the rewritten trigger. The distance-from-the-low "
        "blocker this entry was parked on does NOT reproduce on the ungated parent, and a new and stronger "
        "blocker replaces it. Evidence: a1_c1_tlt_me_entry_ladder.py, a1b, a1c, a1d, a1e."
    ),
    "volatility pop inside": "2026-08-24 verdict: PASS. VIX's 21d rank of 12.7 clears the calm-tape leg but the day's move was -5.50% against the >=+5% pop.",
    "unconfirmed rate rise": "2026-08-24 verdict: PASS. The dollar leg clears easily (DX 21d rank 0.4) and the yield leg does not — a +0.035pt 21-session rise against the +0.20pt floor. The LEVEL variant (yields AT a 52w high) was tested today as a separate candidate and KILLED: it overlaps this rank form on 91% of days, so it is the same object, and the gate-off parent is already negative (h=10 -0.481%, edge -0.942pp). d3_c9_gold_yield_level.py.",
    "rotation gap": "2026-08-24 verdict: PASS. Today's one-day XLV-minus-XLK gap is +1.18pp against the >=+3.0pp trigger.",
    "dollar on a rate rise": "2026-08-24 verdict: PASS. ^TNX's 21d RETURN rank is 49.2 against the >=65 leg; the dollar half is live at rank 0.4 and the rate half is not.",
    "crude through Jackson Hole": "2026-08-24 verdict: PASS twice. The JH-6 anchor was 2026-08-20, four sessions gone, and XLE is -0.17% off its 52w high, which the entry's own second condition forbids.",
    "big up day from inside": "2026-08-24 verdict: PASS. TLT's one-day move was -0.35% against the >=+1.5% thrust rung.",
    "regional banks": "2026-08-24 verdict: PASS. KRE's 5d rank is 6.0 so the breadth state is live again, but the trigger is a cost threshold on history (+0.35% at h=3 ex-crisis) that a new episode cannot move.",
    "high yield across the Jackson Hole": "2026-08-24 verdict: PASS. The JH-5 anchor was 2026-08-21, and the anchor was closed on credit that same morning.",
}

# W12 is rewritten wholesale: new trigger, new blocker, correction recorded.
W12_TRIGGER = (
    "THE MECHANISM, and the entry that stood here before 2026-08-24 was wrong about why. "
    "The anchor itself re-derived cleanly and is still the strongest duration cell measured in "
    "this repo: long TLT MOC nine sessions before the month's last trading-day close, exit at "
    "that close, +0.540% over 288 anchors at t=3.88, month-matched +0.346% at t=3.72, an exit-offset "
    "placebo ladder that SPIKES at the true close (ME+3 +0.065 / ME+0 +0.430 / ME-3 +0.205 / ME-9 "
    "+0.067) rather than the plateau that killed the Jackson Hole and August anchors, holdout "
    "2014-2026 +0.463% at t=3.56 (better than its own in-sample half), top-2 episodes 8% of total, "
    "LOYO and drop-best clean, and it passes the bond-bull fossil test in the modern era. The entry "
    "offset is also flat rather than knife-edged: ME-3 through ME-7 all pay +0.28 to +0.38 "
    "month-matched at t 3.1-4.1, so ME-5 is as good an anchor as ME-9. "
    "CORRECTION, 2026-08-24: the blocker this entry previously carried -- 'forward return regresses "
    "POSITIVELY on distance from the 52w low at +0.126pp per 1% off (t=+2.18)' -- does NOT hold on "
    "the ungated parent it parks. Re-derived, the slope is -0.0082 (t=-0.51) at ME-9 and -0.0064 "
    "(t=-0.60) at ME-5. That gradient lives only on the oversold-GATED subsample (TLT 21d <= -2.5%, "
    "N=50), which this entry's own second debt says to keep OFF and which does not fire today anyway "
    "(TLT 21d -0.95%). Today's +0.86% off the low is uninformative, not adverse: N=7 comparables, "
    "3-4, mean +0.533% but median -0.105%, drop-best -0.045% on 6. "
    "THE REAL BLOCKER, which is stronger: the mechanism has decayed out of the era we trade. The "
    "index-extension story predicts the LAST sessions carry the excess, and TLT's ME-1 -> ME-0 "
    "session ran +25.65 bp at t=3.09 and a 64.3% hit in 2002-2012 against +3.99 bp at t=0.37 and a "
    "48.1% hit vs a 49.3% BASE RATE in 2020-2026 -- replicated on AGG (+10.87 -> +3.77), LQD "
    "(+23.38 -> +3.56) and IEF (+13.19 -> +8.19), with rolling 8-year t falling monotonically from "
    "3.02 to 1.05. The five-session total survives only because a DIFFERENT session carries it in "
    "every era. TURNS ON when TLT's trailing ME-1 -> ME-0 session hit rate is back above its own "
    "base rate (currently 48.1% against 49.3%), or at minimum its trailing 8-year t back above 2. "
    "SECOND condition, independent: NOT August. August ME-5 was 13-for-13 at +1.271% through 2014 "
    "and 5-of-11 at -0.510% since 2015, which is -1.001pp against the rest of the 2015+ cell at "
    "Welch t -1.91, and the pre-2014 vs 2014+ month-profile Spearman is -0.39 (August goes from best "
    "month to 11th of 12). Entry form is already settled if it ever arms: MOC at the ME-5 close, "
    "h=5, exit MOC at the month-end close -- MOC pays +0.4302% at 14.3x cost against a whole-variant "
    "LIMIT(close, -0.25 ATR) at +0.1354% on a 53.8% fill. Tail for sizing: intra-hold worst low vs "
    "entry close averages -1.04%, 5th pctile -3.25%, P(low <= -1 ATR) = 43.8%."
)

NEW = [
    {
        "added": TODAY,
        "title": "The duration-neutral curve position, long IEF against short 0.52 TLT, with the 10-year yield at a 52-week high",
        "cell": "rates, curve x yield-level trigger",
        "trigger": (
            "COST, and it is 8 bps short on the best rung. This is the only cell of a 10-candidate "
            "sweep that cleared its whole battery. Long IEF / short 0.523 TLT (duration-neutral, so it "
            "profits when the long end underperforms the belly per unit of duration) entered MOC when "
            "^TNX closes within 0.25% of its trailing-252 high: rotation permutation over the "
            "PRE-DECLARED 3-vehicle x 2-sign x 5-horizon grid gives P(grid max |t| >= 3.41) = 0.018; "
            "gate attribution ADDS +0.179pp over episode-matched rising-yield days at Welch t +2.00; "
            "era stable AND improving, pre-2018 +0.148% to 2018+ +0.282%; the live midterm half is "
            "+0.299% at an 84% hit and t 2.56; concentration runs the right way at -3% of total (the "
            "top-2 episodes are LOSERS); bootstrap P(mean<=0) 0.007; and it is not a hedge-ratio fit, "
            "since the PIT expanding beta, the rolling-252 beta, the raw sd ratio and a round 2.0 all "
            "land within 2 bps of each other. It needs 30.0 bps to clear the 5x bar on a 6 bps two-leg "
            "round trip and the horizon ladder TOPS OUT at 22.2 bps (h=8, 3.70x), falling to 20.6 bps "
            "= 3.44x under a point-in-time hedge ratio; the full ladder h=1..10 runs 1.12 / 2.12 / "
            "2.02 / 1.95 / 2.58 / 3.00 / 3.21 / 3.70 / 3.45 / 3.68x. The single-leg escape is closed "
            "too: IEF alone at h=10 is +0.140pp of excess = 4.67x. TURNS ON at either (a) the h=8 "
            "episode mean clearing 30 bps, or (b) a two-leg round trip under 4.4 bps -- a real "
            "possibility in Treasury ETFs and the cheaper of the two to check. SECOND BLOCKER that "
            "must clear independently: the episodes whose hold spanned Jackson Hole are 0-for-6, at "
            "-0.465% / -0.406% / -0.629% at h=5/8/10 against JH-out cells of +0.168 / +0.248 / +0.275%. "
            "Note the two OUTRIGHT signs of this trigger are dead and must not be revived: the "
            "falling-yield cell is N=0 by construction, so the trigger says only 'we are in a rising-yield "
            "regime', and the all-days rising-regime control beats the conditional by -0.130pp."
        ),
        "script": "scratch/pitch_checks/2026-08-24/d2b_c5_flattener_charge.py",
        "source": "stand_down",
        "expires": "2027-02-24",
        "note": "Parked 2026-08-24 as the single best cell of a 10-candidate all-kill sweep. Cost is the only thing between it and a pitch.",
    },
    {
        "added": TODAY,
        "title": "The NARROW energy thrust cluster, two or three names at z10 above 2 rather than five",
        "cell": "energy, breadth-of-thrust count",
        "trigger": (
            "THE COUNT ITSELF, and today sat on the wrong side of the crossing. Within the 11-name "
            "energy complex (XLE XOP USO COP CVX VLO OXY SLB EOG HAL WMB), long XLE h=5 by count of "
            "members at z10 >= 2.0 is monotone and crosses zero at four: 2 names +0.715%, 3 names "
            "+0.718%, 4 names +0.139%, 5 names -1.002%. The narrow cell [2,3] pays +0.699% at h=5 over "
            "99 episodes, t=2.40, 63.6% hit, +0.459pp over all days and +0.491pp over its own local "
            "+/-126td control, positive at every horizon 1 through 7. TURNS ON when the live count is "
            "2 or 3; on 2026-08-24 it was 5 (VLO 2.56, COP 2.35, XLE 2.18, XOP 2.10, CVX 2.04) and the "
            "broad form is what was tested and killed. THREE DEBTS before it is ever pitched, none "
            "optional. (1) It is a post-hoc band recovered from the corpse of the broad cell, so the "
            "anti-rescue rule applies and it needs its own pre-registration and a forward re-derivation. "
            "(2) The reference class has not been run on the NARROW form; the broad form ranked 8 of 8 "
            "sectors at P(random sector >= energy) = 1.000, and the narrow form must be shown not to be "
            "an any-sector effect. (3) August is weak even here, +0.246% on 4 episodes against +0.719% "
            "elsewhere. Standing caveat that survives any count: the complex has 1.82 effective names "
            "of 11 (PC1 73.5%), so a count is a coarse read on one factor -- P(XLE thrusting | 5 of 11 "
            "thrusting) is 0.920 -- and any narrow-cluster claim owes the effective-N number beside it."
        ),
        "script": "scratch/pitch_checks/2026-08-24/b4_tail_risk_and_nearmiss.py",
        "source": "near_miss",
        "expires": "2027-02-24",
        "note": "Parked 2026-08-24 as the positive object inside a killed cell, with three explicit debts before it can trade.",
    },
    {
        "added": TODAY,
        "title": "Cross-sectional new-high breadth on a survivorship-free universe, with the index further off its high",
        "cell": "us_large, breadth x index-distance",
        "trigger": (
            "TWO NUMBERS, and both have to move. The survivorship-free 9-sector version clears its "
            "controls where the 218-name tape does not: episodes +0.423% at h=5 against CTRL-a +0.239%, "
            "CTRL-b +0.192% and CTRL-c local +/-126td +0.289%, era-stable at pre-2018 +0.334% and 2018+ "
            "+0.620%, top-2 episodes 15% of total, drop-top-2 +0.388%, midterm +0.441% vs non-midterm "
            "+0.455%, surviving a month-end calendar-position control at +0.066pp over all days sitting "
            "at ME-3..ME-7, at 28x cost. TURNS ON at (a) SPY more than 2.0% below its 52-week high, "
            "where the (-5%,-2.0%] rung pays +0.885% over 38 episodes at t=3.21 -- today was -1.56%, "
            "landing in the (-2%,-0.5%] rung that pays +0.308% -- AND (b) raw-21d fragility at or below "
            "50, today 60.1. Condition (b) is not decoration: split on the live exposure-leg rule the "
            "entire edge is in the complement of today's state (leg-OFF +0.008% at a 50.0% hit against "
            "leg-ON +0.754% at 81.0%, t=3.10), and the trigger population's median dial ma10(63d) is "
            "24.8 with an all-time max of 80.6 against today's 89.5. STANDING BLOCKER that no threshold "
            "fixes: the breadth gradient runs BACKWARDS past two sectors (0 of 9 at a high +0.317%, 2 of "
            "9 +0.310%, 3 of 9 -0.009%, >=5 -0.136%), so this can only ever be a NARROW-breadth cell, "
            "and the breadth leg alone is worth -0.086pp against doing nothing while the index-distance "
            "leg carries the whole cell. Do not re-derive it on the 218-name tape: that universe is "
            "today's survivors, its own PIT percentile read 54.2 on 2026-08-24 against a full-sample "
            "61.2, and the two universes disagree about the gate's worth by an order of magnitude."
        ),
        "script": "scratch/pitch_checks/2026-08-24/c2_c6_breadth_attribution.py",
        "source": "near_miss",
        "expires": "2027-02-24",
        "note": "Parked 2026-08-24. Needs the index deeper off its high AND a calmer dial; the breadth leg is the weak half, not the strong one.",
    },
]


def main() -> None:
    wl = load_watchlist()
    entries = wl.get("entries", [])
    print(f"loaded {len(entries)} active entries")

    # 1. prune the expired entry the state builder flagged
    before = len(entries)
    entries = [e for e in entries
               if "industry-wide five-day breadth washout" not in e.get("title", "")]
    print(f"pruned {before - len(entries)} expired entry")

    # 2. refresh notes with today's verdict; rewrite W12 wholesale
    touched = 0
    for e in entries:
        title = e.get("title", "")
        if "month-end close, ungated" in title:
            e["trigger"] = W12_TRIGGER
            e["expires"] = "2027-08-24"
        for frag, note in VERDICTS.items():
            if frag in title:
                e["note"] = note
                touched += 1
                break
        else:
            print(f"  WARNING no verdict matched: {title[:70]}")
    print(f"refreshed {touched} notes; W12 trigger rewritten")

    # 3. append today's near-misses (idempotent on title)
    have = {e.get("title") for e in entries}
    added = 0
    for n in NEW:
        if n["title"] in have:
            print(f"  already present, skipping: {n['title'][:60]}")
            continue
        entries.append(n)
        added += 1
    print(f"appended {added} near-miss entries")

    wl["entries"] = entries
    wl["expired"] = []
    save_watchlist(wl)
    print(f"saved {len(entries)} active entries -> data/pitch_watchlist.json")


if __name__ == "__main__":
    main()
