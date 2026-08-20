"""Post-publish: stamp today's verdict on every active watchlist entry and
append the three near-misses from the 2026-08-20 stand-down.

Nothing expired and nothing fired, so this is a note refresh plus three adds.
Verdicts come from scratch/pitch_checks/2026-08-20/04_watchlist_verdicts.py and
the surface map's section 5.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WL = ROOT / "data" / "pitch_watchlist.json"
TODAY = "2026-08-20"

# keyed by a distinctive substring of the entry title
NOTES = {
    "Long TLT from the NFP close": (
        "2026-08-20 verdict: PASS, unchanged. The next NFP is 2026-09-04, +11 td, "
        "still past the 10 td horizon cap and still midterm."),
    "Credit-quality divergence, long LQD": (
        "2026-08-20 verdict: PASS, state still live and the count still will not "
        "move. HYG is 0.10% off its 52w high and LQD 1.17% off its low, so both "
        "rungs clear, and the declustered episode count is unmoved at 4 across "
        "[2018, 2026] against the >=8-over->=3-years-ex-2018 arm."),
    "Long SVXY overnight into the CPI print": (
        "2026-08-20 verdict: PASS. Next CPI is 2026-09-11, +15 td; the re-measure "
        "is owed at the 2026-09-10 run. NEW DEBT from today's C4 work, and it "
        "bears directly on this entry's instrument: SVXY's 2018-02-28 leverage "
        "break is worth measuring rather than assuming, at pre-break daily sd "
        "4.56% and worst day -82.96% against 2.32% and -21.43% after. This "
        "entry's 100-event overnight sample spans the break; when it is "
        "re-measured, split it."),
    "Long GLD on a miner-led thrust": (
        "2026-08-20 verdict: PASS, and the divergence is now absent in the other "
        "direction. GDX's 5d rank is 80.2 against the >=95 needed and GLD's is "
        "65.5 against the <95 needed, so today is the two rallying together, "
        "which is the +0.239% cell rather than the +0.832% one. The 2026-08-19 "
        "trend-state debt still stands: GLD is 16.55% below its 52w high with a "
        "63d rank of 34.9, i.e. a bounce inside a drawdown."),
    "Long XLE on a crude one-day thrust": (
        "2026-08-20 verdict: PASS, no pop (USO 1d +0.19%). NEW DEBT from today's "
        "C8 Jackson Hole work, which touched the same complex: 0 of 26 JH anchors "
        "have ever paired XLE within 5% of its 52-week high with a USO 63d rank "
        "under 40, and the one anchor with XLE at its high (2016) was the worst "
        "episode in that sample at -10.97%. XLE-at-the-high is a hostile state "
        "for any long-energy cell, so quote it when this entry next fires."),
    "Long TLT with the whole investment-grade complex": (
        "2026-08-20 verdict: PASS, and the price rung has BROKEN for the first "
        "time in eight sessions. TLT is 2.05% off its 52w low against the <=0.5% "
        "the tight rung needs (IEF 1.31%, LQD 1.17%), so the cluster that ran "
        "2026-08-03 to 08-19 has ended without ever satisfying the freshness leg."),
    "Long SPY on a skew spike alone": (
        "2026-08-20 verdict: PASS. ^SKEW's 5d rank is 89.3 against the >=95 "
        "needed, a wider miss than yesterday's 94.8. The SPY leg still clears "
        "(1.13% off its 52w high against the >1% required) and the midterm leg "
        "is still the structural blocker until 2027."),
    "Fade a crude thrust out of a deep base": (
        "2026-08-20 verdict: PASS. USO's 5d rank is 68.7 against the >=90 "
        "needed; the 63d rank 4.4 leg clears. Post-2020 episode count unchanged "
        "at 4 of 8."),
    "Long the medical-device thrust": (
        "2026-08-20 verdict: PASS, and closer than yesterday. IHI's 21d rank is "
        "99.6 against the 100 required. NEW DEBT: today's bank-breadth work "
        "re-ran the reference-class test this entry parks on and got Cochran Q "
        "6.65 on 11 df with a dispersion ratio of 0.71 across 12 industry groups "
        "- the second independent sample saying no industry or sector label "
        "carries information. The arm condition (Cochran Q p < 0.05 on the "
        "27-ETF class) just got harder to believe, not easier."),
    "Long China's five-day break": (
        "2026-08-20 verdict: PASS. FXI 5d rank 66.7 (needs <=20), 21d rank 74.6 "
        "(needs >=80), EEM 5d -0.53% (needs positive): all "
        "three legs fail. NEW DEBT that bears on the whole country family: "
        "today's C5 permutation over 11 EM/intl vehicles gives P(max name excess "
        ">= the best name's) = 0.283 at h=5 and 0.641 at h=10, so a single "
        "country beating its class is what the null does routinely. Any re-test "
        "of this entry owes that permutation, not just the EEM residual."),
    "An industry-wide five-day breadth washout": (
        "2026-08-20 verdict: CHECK, and the FIRST half-fire since the entry was "
        "written. Banks (JPM BAC C WFC GS MS USB KEY RF STT SCHW) sit at 72.7% of "
        "names with a 5d rank <= 20, clearing the >=70% breadth leg, but the "
        "median 63d rank is 82.5 against the <70 the entry requires, which is "
        "exactly the intact-trend half. Industrials are the mirror: median 63d "
        "rank 65.5 clears, breadth 52.4% does not. The state was taken through a "
        "full check anyway (see the new KRE/XLF entry) and the premise did NOT "
        "replicate outside insurance: on banks the intact-trend half is +0.225% "
        "at h=5 on XLF rather than a -0.789% loser, so there is nothing to "
        "invert. The untested selection leg this entry parks on was also run: "
        "the alphabetical placebo FAILED to kill for the first time in this repo "
        "(signal-picked four beat BAC/C/GS/JPM by +0.589pp at h=3), and the cell "
        "died to its reference class regardless."),
    "Long TLT on the NOVEMBER month-position": (
        "2026-08-20 verdict: PASS, parks to trading days 4-12 of November 2026."),
    "Short SPY with the index at a 52w high while the long end": (
        "2026-08-20 verdict: PASS, and now failing on BOTH legs. SPY is 1.13% off "
        "its 52w high against the <=0.5% required and TLT is 2.05% off its low "
        "against the <=1% required; yesterday only the SPY leg failed. The "
        "de-concentrated 5x-cost trigger is unchanged."),
    "Long TLT into the month-end close": (
        "2026-08-20 verdict: PASS on both legs. Today's close is month-end minus "
        "SEVEN (2026-08-31 is the last session) against the minus-nine the anchor "
        "is defined at, and TLT is 2.05% above its 52w low against the >3% the "
        "trigger needs."),
    "Long SPY on a volatility pop inside an already-calm tape": (
        "2026-08-20 verdict: PASS, and further away than yesterday. The calm leg "
        "clears (VIX 21d rank 22.2, <=25) and the spot leg "
        "clears (SPY +0.21%), but VIX FELL 6.00% on the session against the >=+5% "
        "rise the pop requires. The arm remains a statistical increment (Welch "
        "t >= 2.0 over calm-tape-alone) that no single session can move."),
    "Long gold on an unconfirmed rate rise": (
        "2026-08-20 verdict: PASS, and the two legs have swapped which one is "
        "closer. The dollar leg now clears easily at a DX 21d rank of 0.8 (needs "
        "<=15), and the yield leg fails by more than yesterday: the 21-session "
        "^TNX change is +0.025pt against the +0.20pt required, an eighth of what "
        "it needs. NEW DEBT: today's C6 work establishes that a DX rank extreme "
        "in a quiet year buys a magnitude of -2.32%, the 91.3rd percentile of the "
        "rank<=2 population by depth. This entry's dollar leg is a RANK gate, so "
        "when it next fires beside a live yield leg, quote the DX magnitude too."),
    "Long technology against healthcare after a rotation gap": (
        "2026-08-20 verdict: CHECK, the STATE FIRES and the arm does not. The gap "
        "is 4.57pp (needs >=3.0), SPY is 1.13% off its 52w high (needs within 3%) "
        "and SPY's Wilder-14 ATR is 0.92% of price (needs <1.2%), so all three "
        "legs clear for the first time since the entry was written. The arm fails "
        "on every count: drop-best mean +0.468% against the +0.50% required, N=21 "
        "against >=24, record 12-9 against >=15-9, and today would be the EIGHTH "
        "2026 episode where the arm explicitly requires three new winners landing "
        "OUTSIDE that cluster. Ex-2026 remains 7-7 at -0.599%, and the by-year "
        "table still runs -1.33 (2024) / -1.08 (2025) / +3.86 (2026)."),
    "Short the dollar on a rate rise the currency does not confirm": (
        "2026-08-20 verdict: PASS. ^TNX's 21d rank is 46.8 against the >=65 "
        "needed, and the 21-session level change is +0.025pt. The dollar leg "
        "clears at a 21d rank of 0.8. NEW DEBT, and it is the same one this "
        "entry's own note anticipated: today's C6 work shows the DX rank gate's "
        "entire content is that the trailing year was quiet - rank-extreme but "
        "magnitude-ORDINARY pays +0.162 / +0.214 / +0.638pp at h=3/5/10 while "
        "magnitude-extreme but rank-ordinary pays -0.108 / -0.087 / -0.393pp. "
        "The magnitude-floor restatement this entry already requires is now "
        "mandatory rather than preferred."),
}

NEW = [
    {
        "added": TODAY,
        "title": "Long crude through Jackson Hole, entered six sessions before the conference",
        "cell": "energy x event (jackson_hole)",
        "trigger": "CONCENTRATION, and separately the state XLE is in. This is the "
                   "first event anchor in this repo to SURVIVE the offset placebo "
                   "ladder: long USO at JH-6 ranks 1 of 16 at h=10 (CL=F also 1 of "
                   "16, XLE 3 of 16), pays +2.145% over 20 anchors against an "
                   "anchor-tdom-weighted unconditional August expectation of "
                   "+0.632%, and beats the plain tdom-matched August control of "
                   "+0.593%. It is also not the USO roll artifact that killed the "
                   "thrust-fade family, since USO-minus-CL=F on the anchors is "
                   "+0.049pp against an unconditional roll drag of -0.302pp and "
                   "CL=F carries the same +1.628%. TURNS ON when the drop-best-3 "
                   "h=10 excess reaches +0.50pp: it is -0.056pp today, so three of "
                   "twenty years (2015, 2007, 2021) are the entire effect and "
                   "removing them leaves exactly the unconditional late-August "
                   "window. That needs new winning anchors outside those three "
                   "years, one per August, so this is a multi-year park. SECOND "
                   "leg, and it is a state condition that can change within a "
                   "year: do not take it with XLE within 5% of its 52-week high. "
                   "0 of 26 anchors have ever combined that with a USO 63d rank "
                   "under 40, which is 2026-08-20's state, and the single anchor "
                   "where XLE sat at its high (2016) is the worst episode in the "
                   "sample at USO -10.97% / CL=F -10.49%. Standing caveats: XLE's "
                   "midterm cell is wrong-signed at -0.097% at h=10 while USO's is "
                   "+0.590% against non-midterm +2.664%, so the vehicle and the "
                   "cycle interact; and the pitched USO h=10 cell ranks 2 of 35 in "
                   "the vehicle x horizon grid it came from (excess sd 0.714pp). "
                   "The book has ZERO energy signals in any JH-6..+4 window across "
                   "the 4,741-row ledger, so there is no conflict and no "
                   "confirmation either.",
        "script": "scratch/pitch_checks/2026-08-20/b3b_c8_crude_round2.py",
        "source": "stand_down",
        "expires": "2027-09-01",
    },
    {
        "added": TODAY,
        "title": "Short TLT after a big up day from inside the 52-week low zone",
        "cell": "rates price-state",
        "trigger": "THE PARENT'S SIGN AT THE NEXT RUNG DOWN. The cell as specified "
                   "(TLT 1-day return >= +1.5% while within 4% of its trailing-252 "
                   "low) passed more tests than anything else this morning: +0.638% "
                   "at h=2 over 17 declustered episodes, 13-4, sign p 0.0245, "
                   "bootstrap P(mean<=0) 0.000, local +/-126td control edge "
                   "+0.603pp, tdom-matched +0.702pp, month-matched +0.642pp, LOYO "
                   "floor +0.536%, 21x a 3 bp round trip, worst episode -1.07%, and "
                   "it passes the bond-bull fossil test in reverse (16 of 17 "
                   "triggers are in the rising-yield half by construction, but the "
                   "rising-regime all-days short control is -0.032%, so the regime "
                   "does not explain it). The duration translation is clean rather "
                   "than a TLT quirk: TLT excess +0.673pp against IEF +0.337pp, a "
                   "ratio of 2.00 against a daily-sd ratio of 2.10. TURNS ON when "
                   "the [1.0%, 1.5%) thrust band stops being significantly "
                   "wrong-signed: it pays -0.241% at h=2 over 26 episodes at a "
                   "30.8% hit (sign p 0.986) today, and the loosened 1.0% parent "
                   "loses -0.165% over 33 at a 36.4% hit (sign p 0.960), so the "
                   "positive sign is manufactured by the rung. The full ladder is "
                   "non-monotone and peaks exactly at the pitched value ([1.25,1.5) "
                   "+0.058%, >=1.5% +0.638%, >=1.75% +0.507%, >=2.0% +0.267%). "
                   "SECOND debt, independent of the first: the mechanism has to "
                   "start when the trade does. Per-session increments from the "
                   "entry close run +0.133pp at a 52.9% hit (session +1), +0.505pp "
                   "(session +2), -0.128pp, then -0.432pp at a 29.4% hit (session "
                   "+4), so the whole cell is one non-adjacent session and the "
                   "position hands it back by day four. A short-covering-rally "
                   "story predicts the fade begins at entry. Search priced on the "
                   "125-cell grid actually walked (5 thrust rungs x 5 low rungs x 5 "
                   "horizons, |t| scored so both signs are charged): P(grid max |t| "
                   ">= 3.30) = 0.167. Concentration note for the re-test: 2022 is 7 "
                   "of 17 episodes and ex-2021-2023 leaves N=6 at +0.539%. Today "
                   "2026-08-20 was inside historical support on both dimensions "
                   "(35th percentile by distance from the low, 50th by thrust "
                   "size), so this is a definition problem and not a "
                   "today-is-unusual problem.",
        "script": "scratch/pitch_checks/2026-08-20/a2b_c2_tlt_round2.py",
        "source": "stand_down",
        "expires": "2027-02-20",
    },
    {
        "added": TODAY,
        "title": "Short regional banks against the big-bank index on a breadth washout",
        "cell": "financials, intra-sector breadth",
        "trigger": "COST once the crisis years come out. Short KRE against long XLF "
                   "on the bank-breadth washout (>=70% of an 11-name complex at a "
                   "5d rank <= 20 with the median 63d rank intact) pays +0.702% at "
                   "h=3 on 16-9 and +1.370% at h=10 on 16-9, and beta-neutralising "
                   "does not explain it (KRE-on-XLF beta 0.97/0.98, residual "
                   "+0.698%/+1.353%). TURNS ON at 5x cost ex-crisis: dropping "
                   "2008/2009/2020 leaves +0.102% at h=3 over 18 episodes = 1.5x a "
                   "7 bp two-leg round trip, and +0.114% at h=10 = 1.6x, so the "
                   "trigger is +0.35% at h=3 on the ex-crisis subset. Era split "
                   "runs +1.219% pre-2018 (t 2.25) against +0.043% after (t 0.06) "
                   "and top-2 episodes are 64% of the h=3 total, so this needs new "
                   "modern episodes rather than a wider net. TWO PASSES worth "
                   "keeping, both unusual: the tape over-selection check runs the "
                   "RIGHT way for once (trigger days sit below SPY's 200d 20.8% of "
                   "the time against a 25.4% base rate), and the alphabetical "
                   "placebo FAILED to kill for the first time in this repo - the "
                   "signal-picked four names beat BAC/C/GS/JPM by +0.589pp at h=3 "
                   "and +1.333pp at h=10 market-relative. STANDING BLOCKER that "
                   "survives any trigger: the outright short is dead (short KRE "
                   "ex-2008/2009/2011/2020 is -0.950% at h=10 on 5-12) and so is "
                   "the XLF short (-0.9 / -22.5 / -51.1 bps at h=3/5/10), so if "
                   "this ever arms it arms as the PAIR. And the reference class "
                   "says the bank label carries nothing: across 12 industry groups "
                   "Cochran Q is 6.65 on 11 df with a dispersion ratio of 0.71 and "
                   "P(max group excess >= banks) = 0.761, so any re-test has to "
                   "show the pair form works on OTHER industries too or explain why "
                   "regionals-versus-money-centres is structurally special.",
        "script": "scratch/pitch_checks/2026-08-20/a3c_c7_kre_pair_teardown.py",
        "source": "stand_down",
        "expires": "2027-02-20",
    },
]

wl = json.loads(WL.read_text(encoding="utf-8"))
entries = wl["entries"]

stamped = 0
for e in entries:
    for key, note in NOTES.items():
        if key in e["title"]:
            e["note"] = note
            stamped += 1
            break
    else:
        print("NO NOTE MATCHED:", e["title"][:70])

titles = {e["title"] for e in entries}
added = 0
for n in NEW:
    if n["title"] in titles:
        print("already present:", n["title"][:60])
        continue
    entries.append(n)
    added += 1

wl["entries"] = entries
wl.setdefault("expired", [])
WL.write_text(json.dumps(wl, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"stamped {stamped} notes, added {added} entries, {len(entries)} active, "
      f"{len(wl['expired'])} expired")
