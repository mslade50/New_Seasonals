"""Post-publish watchlist maintenance for 2026-08-13.

Appends today's two near-misses, and stamps the verdicts the B1 surface map
owed every active entry. Nothing fired today, so nothing is pruned; W1 gains
the month-of-year control finding it has owed since 2026-08-10.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import load_watchlist, save_watchlist  # noqa

wl = load_watchlist()
entries = wl.get("entries", [])
by_title = {e["title"]: e for e in entries}

NOTES = {
    "Long TLT from the NFP close to +3td, with the long end at its 52w floor":
        "2026-08-13 verdict: PASS, trigger unchanged - the next NFP, 2026-09-04, "
        "is still a midterm print and is 16 td out besides. THE OWED TDOM DEBT IS "
        "NOW PART-SETTLED, and it is worse than a tdom effect: TLT's 10td lag-1 "
        "forward return has strong MONTH-OF-YEAR structure (Nov +1.059%, Aug "
        "+0.494%, Jun +0.498%, Jul +0.451% against Oct -0.432%, Sep -0.220%, Apr "
        "-0.240%), large enough to swallow a +1% conditional mean whole. A "
        "September NFP entry sits in TLT's second-worst month. Re-derive against "
        "BOTH a month-of-year and a tdom control before trading this in 2027. "
        "(scratch/pitch_checks/2026-08-13/b1b_c4a_round2.py)",
    "Credit-quality divergence, long LQD against short HYG at joint 52w extremes":
        "2026-08-13 verdict: PASS, unchanged. The joint state is still live (HYG "
        "0.00% off its 52w high, LQD 0.75% off its 52w low) but it is still the "
        "cluster that began 2026-07-22, so the count is still 4 episodes with "
        "three of them in 2018.",
    "Long SVXY overnight into the CPI print, MOC the eve to MOO on the print":
        "2026-08-13: the re-measure this entry asks for after each CPI print is "
        "now DUE (CPI printed 2026-08-12) and was deliberately not spent today, "
        "because the next CPI is 2026-09-11, 20 td out, and no legal pitch "
        "horizon reaches it. Owed at the 2026-09-10 run. Unrelated but adjacent "
        "evidence from today: SVXY's short-vol carry is a LAGGING marker - on "
        "98th-percentile contango triggers its trailing 21d return has a median "
        "of +10.46%, so any SVXY entry gate wants the run checked before the "
        "state.",
    "Long GLD on a miner-led thrust the metal has not joined":
        "2026-08-13 verdict: PASS, trigger not live - the required DIVERGENCE is "
        "absent, with GDX's 5d rank at 88.9 against the >= 95 needed and GLD's at "
        "84.9, and the live GDX leg to 08-17 still fails the third condition. "
        "Reinforced today from the event side: GLD across the Jackson Hole run is "
        "-1.213% at 1-4 in midterm years, and GLD/GDX daily correlation now "
        "measures +0.831, above the +0.724 that killed the GLD leg on 08-11.",
    "Long XLE on a crude one-day thrust in the 5 to 6 percent band":
        "2026-08-13 verdict: PASS, trigger not live. USO's one-day move is "
        "-0.24%, which is not a pop in any band.",
    "Long TLT with the whole investment-grade complex pinned at 52w lows":
        "2026-08-13 verdict: PASS, price rung ON and the freshness leg FAILS. "
        "TLT 0.23% / IEF 0.86% / LQD 0.75% are all inside the tight tolerances, "
        "but the episode began 2026-08-03 so today is 7 sessions deep against a "
        "trigger that needs the first trigger day in >= 10 sessions. Add the "
        "month-of-year caveat from W1 before this trades: August is TLT's "
        "second-best month at +0.494%/10td, so a summer reading of this cell "
        "owes a month control too.",
    "Long SPY on a skew spike alone, no volatility condition":
        "2026-08-13 verdict: PASS, trigger not live and both arming legs fail. "
        "^SKEW's 5d rank is 74.2 against the 95 required, SPY is 0.10% below its "
        "52w high against the >1% required, and 2026 is a midterm year.",
    "Fade a crude thrust out of a deep base with a macro print inside the hold":
        "2026-08-13 verdict: PASS, still 4 post-2020 episodes against the 8 "
        "required, and there is no thrust to fade today (USO 1d -0.24%).",
}

for title, note in NOTES.items():
    if title in by_title:
        by_title[title]["note"] = note
    else:
        print(f"WARN: watchlist entry not found, note not applied: {title}")

NEW = [
    {
        "added": "2026-08-13",
        "title": "Long the medical-device thrust, IHI at a 21d rank of 100 out of a drawdown",
        "cell": "sectors price-state",
        "trigger": "the REFERENCE CLASS, plus freshness. Within IHI the cell looks "
                   "like a survivor: h=5 +1.499% over 16 episodes at 12-4, excess "
                   "+1.267pp over its own drift, bootstrap P(mean<=0) 0.0022, "
                   "positive in 9 of 9 firing years and in both eras (pre-2018 "
                   "+1.624%, 2018+ +1.338%), monotone in the rank gate and flat "
                   "across the drawdown gate. It dies to the cross-section: run the "
                   "identical rule on 27 sector ETFs and Cochran Q is 24.56 on 26 df "
                   "(p 0.544, I-squared 0.0%) with a fixed-effect common excess of "
                   "-0.035pp, while permuting the same estimator produces a MAXIMUM "
                   "of +1.92pp from pure noise against IHI's +1.211pp, i.e. "
                   "family-wise p 0.9330 and a below-median best-of-27 draw. TURNS "
                   "ON when the 27-ETF reference class shows real heterogeneity "
                   "(Cochran Q p < 0.05) with IHI's excess above the permutation "
                   "max, AND the state fires on an episode-FIRST day: 15 of the 16 "
                   "measured episodes are depth-1 entries. Two things to fix before "
                   "any re-test: min_gap 21 declustering already takes the excess to "
                   "+0.484pp at bootstrap 0.194, and the headline +13.94% thrust was "
                   "a denominator roll (ret21 jumped 4.90pp on a session price moved "
                   "+0.18%, because the 21-day reference rolled off a -4.13% bar), so "
                   "quote a magnitude gate rather than a rank next time. The FADE "
                   "direction is separately and permanently dead at -0.953% on 5-15.",
        "script": "scratch/pitch_checks/2026-08-13/r1b_multiplicity_max_of_k.py",
        "source": "stand_down",
        "expires": "2027-02-13",
    },
    {
        "added": "2026-08-13",
        "title": "Long China's five-day break inside an intact thrust, FXI while EEM holds",
        "cell": "international price-state",
        "trigger": "the residual, then the definition. The tight cell (FXI 5d rank "
                   "<= 20 while its 21d rank >= 80 and EEM's 5d return is positive) "
                   "is 5-0 at h=3 at +0.834%, sign p 0.0312 against a coin and "
                   "0.0388 against FXI's own 52.2% base rate, but the residual "
                   "against EEM at the measured 1.025 beta is -0.277% because EEM "
                   "paid +1.084% on the same windows, so the outright is EM beta "
                   "with a China label. TURNS ON when the residual against EEM is "
                   "positive across >= 8 declustered episodes AND the rank-5 cut "
                   "survives loosening to 25; today the residual is -0.277% and the "
                   "25 cut pays -0.003% (30 pays -0.426%). The as-specified joint "
                   "state has 4 declustered episodes with three inside one two-month "
                   "window in 2006-07, and the EEM-positive gate puts SPY below its "
                   "200d on 0.0% of trigger days against a 19.7% base rate, so a "
                   "re-test must also show the gate is not simply excluding bear "
                   "tape.",
        "script": "scratch/pitch_checks/2026-08-13/c8b_fxi_tight_teardown.py",
        "source": "stand_down",
        "expires": "2027-02-13",
    },
]

titles = {e["title"] for e in entries}
for entry in NEW:
    if entry["title"] in titles:
        print(f"already present, skipped: {entry['title']}")
    else:
        entries.append(entry)

wl["entries"] = entries
save_watchlist(wl)
print(f"watchlist: {len(entries)} active, {len(wl.get('expired', []))} expired")
for e in entries:
    print(f"  {e['added']}  {e['title'][:70]}")
