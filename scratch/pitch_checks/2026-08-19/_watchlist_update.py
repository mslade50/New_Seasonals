"""Post-publish watchlist maintenance for 2026-08-19.

Appends the three stand-down near-misses and stamps today's verdict note onto
every active entry (stage B1 owed each one a verdict; this records it so
tomorrow's map starts from the answer rather than re-deriving it).
"""
import json
from pathlib import Path

WL = Path('data/pitch_watchlist.json')
w = json.loads(WL.read_text(encoding='utf-8'))
entries = w['entries']

# ---- today's verdict notes, keyed by the entry's `added` + a title fragment
NOTES = {
    ('2026-08-07', 'Long TLT from the NFP close'):
        "2026-08-19 verdict: PASS, unchanged and still structurally unreachable. "
        "Next NFP 2026-09-04 is +12 td, past the 10 td horizon cap, and midterm.",
    ('2026-08-10', 'Credit-quality divergence'):
        "2026-08-19 verdict: PASS, state still live and still one cluster. HYG "
        "0.33% off its 52w high, LQD 0.48% off its low, so the count is unmoved "
        "at 4 declustered episodes with three of them in 2018.",
    ('2026-08-11', 'Long SVXY overnight into the CPI print'):
        "2026-08-19 verdict: PASS. Next CPI is 2026-09-11, +17 td; the re-measure "
        "is owed at the 2026-09-10 run.",
    ('2026-08-11', 'Long GLD on a miner-led thrust'):
        "2026-08-19 verdict: PASS, divergence still absent (GDX 5d rank 34.1, GLD "
        "33.3, against the >=95 / <95 split). NEW DEBT from today's macro work, "
        "and it corrects a premise this entry leans on: 'gold is hot' is a 21-day "
        "read only. GLD's 63d rank is 30.6, it closed BELOW its 200d (398.55 vs "
        "412.74) and it is -19.6% off its 252d high, i.e. a bounce inside a "
        "drawdown. When this entry next fires, quote the trend state beside the "
        "rank before building any selection argument on either.",
    ('2026-08-11', 'Long XLE on a crude one-day thrust'):
        "2026-08-19 verdict: PASS, no pop (USO 1d +0.28%). NEW DEBT that changes "
        "which series this entry may read: CL=F and USO agree on a 63d-rank gate "
        "on 95.7% of 4,805 shared days, and today is in the disagreeing 4.3% "
        "(CL=F 21.8 against USO 6.0) because of USO roll decay. This entry's "
        "trigger is a one-day pop where the two agree, but any crude STATE gate "
        "attached to it must be read off the front month.",
    ('2026-08-12', 'Long TLT with the whole investment-grade complex'):
        "2026-08-19 verdict: PASS. The tight price rung is live for a seventh "
        "straight session (TLT 0.38% off its 52w low, IEF 0.83%, LQD 0.48%) and "
        "the freshness leg fails by more than ever: cluster depth 7 against the "
        ">=10 sessions of separation required. The 2026-08-18 distance gradient "
        "still argues against the rung itself.",
    ('2026-08-12', 'Long SPY on a skew spike alone'):
        "2026-08-19 verdict: PASS, and the closest this entry has come. ^SKEW 5d "
        "rank 94.8 against the >=95 needed, a miss of 0.2. The SPY leg CLEARS for "
        "the first time (1.34% off its 52w high against the >1% required). The "
        "year is still midterm, which is the structural leg and cannot change "
        "until 2027.",
    ('2026-08-12', 'Fade a crude thrust out of a deep base'):
        "2026-08-19 verdict: PASS. USO 5d rank 65.1 against the >=90 needed; the "
        "63d rank 6.0 leg clears. Post-2020 episode count unchanged at 4 of 8.",
    ('2026-08-13', 'Long the medical-device thrust'):
        "2026-08-19 verdict: PASS. IHI 21d rank 99.2 against the 100 required. "
        "NEW DEBT: today's base-breakout cross-section found the t-63 roll-off "
        "dominates the day's own move on 61.8% of trigger name-days on the "
        "rank-HIGH tail, against 37.3% on the rank-LOW one measured 08-18. This "
        "entry already owes a magnitude gate rather than a rank; that debt just "
        "got larger, since a 21d rank of 100 is the same class of statistic.",
    ('2026-08-13', "Long China's five-day break"):
        "2026-08-19 verdict: PASS. FXI 5d rank 29.0 (needs <=20), 21d rank 57.1 "
        "(needs >=80) and EEM 5d -0.14% (needs positive) all fail. The "
        "residual-positive condition this entry survives on has still never been "
        "met.",
    ('2026-08-14', 'An industry-wide five-day breadth washout'):
        "2026-08-19 verdict: PASS. No coherent industry has >=70% of its names at "
        "a 5d rank <=20 with a median 63d rank below 70; the semis complex is "
        "washed on 63d but its 5d ranks span 4 to 64. NEW DEBT: the alphabetical "
        "placebo went 5-for-5 today on the megacap-washout cross-section "
        "(+0.122% for the signal against +0.342% for the alphabetically-first, "
        "market-relative), so the untested selection leg this entry parks on is "
        "now the likeliest thing to kill it. Run it first.",
    ('2026-08-17', 'Long TLT on the NOVEMBER month-position effect'):
        "2026-08-19 verdict: PASS, parks to trading days 4-12 of November 2026.",
    ('2026-08-17', 'Short SPY with the index at a 52w high'):
        "2026-08-19 verdict: PASS. The TLT leg clears (+0.38% off its 52w low); "
        "the SPY leg fails at 1.34% off its high against the <=0.5% required, and "
        "has moved further away since yesterday. The de-concentrated 5x-cost "
        "trigger is unchanged.",
    ('2026-08-18', 'Long TLT into the month-end close, ungated'):
        "2026-08-19 verdict: PASS on both legs. TLT is +0.38% above its 52w low "
        "against the >3% the trigger needs, and today is month-end-minus-EIGHT "
        "(2026-08-31 is the last session) rather than the minus-nine the anchor "
        "is defined at, where the headline halves.",
    ('2026-08-18', 'Long SPY on a volatility pop inside an already-calm tape'):
        "2026-08-19 verdict: PASS, and the nearest miss on the tape. The calm leg "
        "clears (VIX 21d rank 17.1 <=25) and the spot leg clears (SPY -0.68% on "
        "the day, inside the 0.75% bound), but the pop is +4.28% against the >=5% "
        "required. The arm condition is a statistical increment (Welch t >=2.0 "
        "over calm-tape-alone) that no single session can move.",
}

stamped = 0
for e in entries:
    for (added, frag), note in NOTES.items():
        if e['added'] == added and frag in e['title']:
            e['note'] = note
            stamped += 1
            break
    else:
        print(f"  UNSTAMPED: {e['added']} {e['title'][:60]}")

print(f"stamped {stamped}/{len(entries)} existing entries")

# ---- today's three near-misses, from the stand-down's `closest` block
NEW = [
    {
        "added": "2026-08-19",
        "title": "Long gold on an unconfirmed rate rise, with both dials at force",
        "cell": "rates x dollar_fx -> gold",
        "trigger": "the YIELD MAGNITUDE, not the rank. The pitched rank form is "
                   "dead (knife edge at exactly 21 sessions: the 10/13/15/18-day "
                   "lookbacks are all wrong-signed at h=1, the sign flips between "
                   "the 15 and 18-day forms of the identical rule, and the "
                   "wrong-signed side is better populated at N=52 against 47). "
                   "The one cell with both dials at force survives that ladder: "
                   "DX 21d rank <= 15 AND a 21-session yield rise >= +0.20pt pays "
                   "+0.551% at h=1 over 16 episodes, 12-4, sign p 0.038, and "
                   "+1.126% at h=3 (t=2.19). TURNS ON when both legs are live "
                   "simultaneously; today clears the dollar at rank 14.3 and "
                   "misses the yield leg by half, +0.108pt against +0.20pt. Two "
                   "debts before it ever trades: it is search-contaminated at "
                   "P(grid max t >= 2.06) = 0.937 over the 168-mask grid it came "
                   "out of, so it needs re-deriving forward rather than from that "
                   "grid; and the parent's dose response runs BACKWARDS (+0.432 / "
                   "+0.384 / +0.225% at yield floors of +0.05 / +0.10 / +0.20pt), "
                   "so the forced-dial cell has to explain why it is the exception "
                   "to its own parent's gradient rather than the top of it. "
                   "Standing pass worth keeping: the trigger is NOT a gold-bull "
                   "selector (GLD above its 200d on 70.2% of trigger episodes "
                   "against a 66.1% base rate) and day-1-of-run entries pay "
                   "+0.543% at t=2.89 against -0.664% at day 16+, so freshness is "
                   "on this cell's side rather than against it.",
        "script": "scratch/pitch_checks/2026-08-19/c7f_h1_and_reopen.py",
        "source": "stand_down",
        "expires": "2027-02-19",
        "note": "Recorded with its passes as well as its blocker. It cleared the "
                "bond-bull fossil test (the secular rising-yield half, which is "
                "today's, pays +0.498% at a 70.6% hit against +0.119% falling), "
                "gate attribution both ways at h=1 and h=3 (joint +0.570 against "
                "TNX-alone +0.216 and DX-alone -0.124), and the mid-cluster trap "
                "does not apply since the joint trigger turned on today. It is "
                "the first cross-asset macro cell in this repo to pass the fossil "
                "test and still die."
    },
    {
        "added": "2026-08-19",
        "title": "Long technology against healthcare after a rotation gap, in calm near-high tape",
        "cell": "sectors, one-day rotation extreme",
        "trigger": "NEW EPISODES OUTSIDE THE 2026 CLUSTER, and the arithmetic is "
                   "exact. The subclass matching today (one-day XLV minus XLK gap "
                   ">= 3.0pp, SPY within 3% of its 52w high, SPY Wilder-14 ATR "
                   "under 1.2% of price) pays +0.889% at h=3 over 21 episodes, and "
                   "the rotation trigger genuinely beats an ignorant 'any big "
                   "tech-down day in the same tape' placebo (-0.066 / +0.036 / "
                   "-0.164 / +0.321 at h=1/2/3/5), which is the one test it "
                   "passed. It dies on concentration: the top 2 episodes are 96% "
                   "of the h=3 total and both are 2026 prints, ex-2026 is 7-7 at "
                   "-0.599% and negative at 5 of 6 horizons, and the by-year table "
                   "runs -1.33 (2024) / -1.08 (2025) / +3.86 (2026). TURNS ON when "
                   "the h=3 subclass drop-best-episode mean reaches +0.50% at "
                   "N >= 24 with a record of 15-9 or better, which needs three new "
                   "subclass episodes, all three winners, landing OUTSIDE the "
                   "current 2026 rotation cluster. The state fires roughly 7x a "
                   "year now, so this is measurable within months. Two caveats "
                   "that survive any trigger: the apparent midterm-year "
                   "conditioner (+3.553% against -0.751%) is that same 2026 "
                   "cluster relabelled at 7 of its 8 episodes, so do not read it "
                   "as a cycle effect; and the NAKED long XLK beats the pair at "
                   "h=5 and h=7, so if this ever arms it arms as an outright, not "
                   "as a pair.",
        "script": "scratch/pitch_checks/2026-08-19/a5b_c2_dropbest.py",
        "source": "stand_down",
        "expires": "2027-02-19",
        "note": "Method note attached to the whole family: the headline '99.3rd "
                "percentile' one-day gap is a FULL-SAMPLE rank and therefore "
                "lookahead; the point-in-time trailing-252d rank is 97.2 and does "
                "not clear the >=99 threshold the rank-based version was built on. "
                "Only the absolute-pp definitions fire, which is why this entry is "
                "specified in pp. Also: pitch_lab.pct_rank ranks the n-day PERCENT "
                "CHANGE, so it is meaningless on a spread series that crosses "
                "zero - do not reach for it here."
    },
    {
        "added": "2026-08-19",
        "title": "Short the dollar on a rate rise the currency does not confirm",
        "cell": "rates x dollar_fx",
        "trigger": "COST on an honest magnitude floor. The rank form (TNX 21d "
                   "rank >= 65 while DX 21d rank <= 20) pays +0.234% at h=5 over "
                   "56 episodes at Welch t +1.68 against its own drift, but the "
                   "exact sign test never clears 0.25 at any gap or horizon, and "
                   "the 21-session lookback is the single positive point in a "
                   "ladder whose 10/13/15/18-day neighbours are all negative at "
                   "both h=3 and h=5 with MORE observations. Restated as a "
                   "magnitude floor rather than a rank, TNX 21d level rise >= "
                   "+0.10pt pays 3.9 bps at h=5, which is 2.6x the 1.5 bp "
                   "DX-futures round trip. TURNS ON at 5x cost, i.e. 7.5 bps on "
                   "the magnitude-floor form. Structural blocker to fix first: "
                   "the thesis is entirely a size-of-repricing story and neither "
                   "dial has a dose response (yield-rise buckets +0.289 / +0.119 "
                   "/ +0.253%, dollar-fall buckets +0.816 / +0.225 / +0.200%, both "
                   "non-monotone with the middle bucket weakest), so a bigger "
                   "sample alone will not rescue it - the mechanism has to be "
                   "restated. Search-priced at P(grid max t >= 1.73) = 0.810.",
        "script": "scratch/pitch_checks/2026-08-19/c6e_lookback_fullpanel.py",
        "source": "stand_down",
        "expires": "2027-02-19",
        "note": "Vehicle finding worth keeping independently of this entry: "
                "UUP-versus-DXY-spot is a COST problem and not a signal problem. "
                "Matched episodes give short-DXY +0.131% against short-UUP "
                "+0.118%, a gap of 1.3 bps at t=0.55 with 95.5% sign agreement, "
                "and the all-days structural gap is 1.4 bps per 5td. The "
                "registry's standing 'UUP is dead' entry stands on drag alone; do "
                "not also claim the two vehicles disagree about the effect. Era "
                "decay is real but is not the headline (2018+ pays 5.1 bps = 3.4x "
                "cost at a 43.8% hit), and the cell PASSES the bond-bull fossil "
                "test."
    },
]

have = {(e['added'], e['title']) for e in entries}
for n in NEW:
    if (n['added'], n['title']) in have:
        print(f"  already present: {n['title'][:60]}")
    else:
        entries.append(n)
        print(f"  appended: {n['title'][:60]}")

# ---- prune: nothing expired today, but run the check so it is not assumed
TODAY = '2026-08-19'
live, dead = [], w.get('expired', [])
for e in entries:
    if e.get('expires') and e['expires'] < TODAY:
        dead.append(e)
        print(f"  EXPIRED: {e['title'][:60]}")
    else:
        live.append(e)
w['entries'] = live
w['expired'] = dead

WL.write_text(json.dumps(w, indent=1) + '\n', encoding='utf-8')
print(f"wrote {WL}: {len(live)} active, {len(dead)} expired")
