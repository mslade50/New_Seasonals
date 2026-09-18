"""Post-publish watchlist maintenance for 2026-09-08.

Retires entry 40 (ITA) because its arm RAN and returned P=0.6892; amends entry
33 rather than minting a duplicate for the second-print SVXY rung; appends the
two genuinely new near-misses; prunes anything past its expiry.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "data/pitch_watchlist.json"
ASOF = "2026-09-08"

w = json.loads(P.read_text(encoding="utf-8"))
entries = w["entries"]
expired = w.setdefault("expired", [])

# --- 1. RETIRE the ITA entry: its arm ran and failed, so it is not re-parked.
retired = [e for e in entries if e["title"].startswith("Long ITA at a 21-day rank floor")]
assert len(retired) == 1, f"expected 1 ITA entry, found {len(retired)}"
ita = retired[0]
ita["note"] = (
    "RETIRED 2026-09-08. The arm RAN and failed. The 13-name subsector reference "
    "class (ITA IHI IBB XBI ITB XHB XRT XME XOP OIH KRE SMH IYR) puts ITA's "
    "excess at a max-of-13 permutation P of 0.6892 at h=10 excess, 0.9771 at "
    "h=10 residual, 0.9889 at h=5 excess and 1.0000 at h=5 residual, against an "
    "arm requiring P <= 0.10, and on every one of the four bases ITA's observed "
    "excess sits BELOW the null's median best-of-13 draw. The class is "
    "homogeneous (Cochran Q 7.80-14.21 on 12 df, I-squared 0.0-15.6%) with a "
    "fixed-effect common excess of +0.020pp at h=10; IHI beats ITA outright at "
    "h=10 (+0.959 vs +0.869pp) and XRT beats it at h=5 (+0.782 vs +0.344pp). The "
    "effect is class-wide and ITA was the top draw of thirteen correlated "
    "subsectors selected for being the day's outlier, which is the shape that "
    "closed the country-decoupling family at P 0.477. SEPARATELY, the entry's "
    "published numbers do not reproduce: N=43/+0.543%/bootP 0.071 at h=5 and "
    "N=29/+1.223%/bootP 0.010 at h=10 clean to N=42/+0.629% and N=28/+1.443%, "
    "the difference being one phantom episode dated 2026-09-02 minted by "
    "_survey_lib.align() forward-filling a FORWARD-RETURN series, whose booked "
    "-4.942% at h=10 is exactly the 'worst episode -4.94%' the entry quotes. Do "
    "not revive below P 0.05. (c12_ita_refclass.py, c12b_repro_discrepancy.py)"
)
entries.remove(ita)
expired.append(ita)

# --- 2. AMEND entry 33 instead of minting a duplicate anchor.
amend = [e for e in entries if e["title"].startswith("Long SVXY into a scheduled print out of a 21-day VIX range")]
assert len(amend) == 1, f"expected 1 entry-33 match, found {len(amend)}"
amend[0]["note"] = (
    "2026-09-08 verdict: PASS, and the entry gains a SECOND RUNG rather than a "
    "duplicate entry. Today is the PPI k=-2 anchor at runway 1, which this entry "
    "disqualifies by name, and the compression reading is 4.37 on the 2026-09-04 "
    "bar (trailing-252 basis), still 0.63 percentile points below the (5,15] "
    "band's lower edge for a fourth session running (3.57 -> 3.97 -> 4.37 -> "
    "4.37, unchanged across the closure because there were no new bars). "
    "AMENDMENT, from 2026-09-08's C3 work: a distinct and better anchor exists on "
    "the SAME session this entry already names. Long SVXY entered MOC on the "
    "SECOND print of a back-to-back pair, held 5 sessions into the cleared "
    "calendar, pays +1.268% over 23 live-era (-0.5x) episodes at an 82.6% hit, "
    "alpha +0.793% against a trigger-set beta, sign p 0.017, LOYO minimum "
    "+0.908% -- and unlike the pre-print form the segment carrying the return IS "
    "the segment being traded. It needs BOTH of the following and has neither: "
    "(1) the VIX 21-day relative-range percentile closing in (5.0, 15.0] on the "
    "entry session, 4.37 today; and (2) the live-era post-pair cleared-calendar "
    "segment (print2 -> +2 sessions) recovering above +0.50%, against +0.117% at "
    "a 56.5% hit on n=23 now. Its next instance is the 2026-09-11 CPI close, "
    "which is this entry's own next qualifying anchor -- grade both rungs that "
    "morning, do not mint a second entry. ALSO SETTLED 2026-09-08, do not re-run: "
    "'a print on the very next session' is SET-IDENTICAL to runway == 1 (N=139 "
    "vs N=139), so a back-to-back pair can never be anything but this entry's "
    "dead half, and the pre-pair hold is wrong-signed on the unlevered vehicle "
    "(short ^VIX cumulates -0.52 / -1.36 / -0.67 / -0.45 / +0.92 through it). "
    "(c3_pair_zero_runway_vol.py, c3b, c3c, c3d_live_era_alpha_loyo.py)"
)

# --- 3. Append the two genuinely new near-misses.
entries.append({
    "added": ASOF,
    "title": ("Short IEF at h=5 with the commodity complex at a 252-day high "
              "and an inflation print inside the hold"),
    "cell": "commodities price-state x event -> rates, the duration expression",
    "trigger": (
        "THE PLACEBO LADDER, which is a mechanism test rather than a level, and "
        "note this entry is the one leg of a three-candidate energy sweep that "
        "did not die. The cell: 49 episodes, +0.184%, 63.3% hit, sign p 0.0427, "
        "6.1x cost, and its print gate contributes +0.173pp at EPISODE level and "
        "+0.325pp at DAY level -- the same sign at both, which is precisely the "
        "signature the 2026-09-07 anchor-swap trap lacks. It also holds with the "
        "inflation-shock years removed: +0.255% on n=20 at a 70.0% hit, sign p "
        "0.0577, so it is not the 2007/2008/2021/2022 artefact that killed the "
        "long-commodity form beside it (there, four years held 26 of 53 episodes "
        "and MORE than 100% of the total). WHAT KILLED IT: the true print anchor "
        "ranks 5 of 11 on its own k=-5..+5 placebo ladder, with k=-5 at +0.262% "
        "and k=-3 at +0.236% both paying more than the true anchor's +0.184%. "
        "Until the anchor outranks its own placebos this is short-duration "
        "momentum wearing an event label, and with the whole IG complex already "
        "pinned at 52-week lows that momentum is free. TURNS ON when the true "
        "print anchor ranks 1st or 2nd of 11 on the k=-5..+5 placebo ladder by "
        "episode mean. Two standing notes if it ever arms: the TLT expression is "
        "NOT the same trade (its pitched h=3 form is +0.072% at 2.4x cost with "
        "the print gate SUBTRACTING at episode level, placebo rank 8 of 11), and "
        "a fully-live TLT three-way at h=8 that reads 5-for-5 at +1.670% with "
        "placebo rank 1 of 11 was tested and killed as a lucky subset -- it "
        "discards 49 of 54 episodes while the complement still pays +0.379% "
        "against the parent's +0.443%, P(random 5-subset beats it) 0.0631, "
        "family-wise P 0.648 over the 16 cells searched. Do not rediscover it."
    ),
    "script": "scratch/pitch_checks/2026-09-08/c8b_print_gate_and_regime.py",
    "source": "near_miss",
    "expires": "2027-03-08",
    "note": "",
})

entries.append({
    "added": ASOF,
    "title": ("Long SPY across the September PPI-then-CPI pair, entered two "
              "sessions before the first print"),
    "cell": "back-to-back inflation print pair x month-of-year, us_large",
    "trigger": (
        "THE PARENT'S SIGN, and read the scan charge before touching this. The "
        "September subcell is 14 observations at +0.844% and an 85.7% hit; it "
        "survives the decomposition that kills the annual cell (today's bucket, "
        "FOMC OUTSIDE the hold, is 11 at +0.518%, 81.8%, sign p 0.033) and it "
        "survives month-by-trading-day-of-month matching at +0.755pp. That is "
        "more than the parent manages. WHAT KILLED IT is that it is a one-in-six "
        "month scan sitting on an INVERTED parent: P(some month with n>=3 looks "
        "this good) = 0.1618 across the 12-month scan, and the gate it decorates "
        "is worth -0.115pp where its own complement (PPI with NO CPI on the next "
        "session) pays +0.024pp, so the pair gate selects its parent's worse "
        "half. TURNS ON only when BOTH hold: the 12-month permutation P falls "
        "below 0.05, AND the parent pair gate stops being negative, i.e. the "
        "back-to-back cell's excess over the single-PPI complement turns "
        "positive. Re-measure after each new PPI-then-CPI pair; September pairs "
        "accrue at roughly one a year. STANDING CORRECTION filed the same "
        "morning: the surface map's claim that the PPI-then-CPI ordering was "
        "unmeasured is FALSE -- it reproduces the 2026-08-10 line at -0.1135% on "
        "N=127 (registry -0.071% on N=133), so today's ordering is the NEGATIVE "
        "side of a pair this repo already measured, and the annual cell should "
        "never be re-opened as novel. Also settled: the placebo ladder ranks the "
        "true anchor 8 of 11 on SPY and 9 of 11 on IWM, with k=+1 paying +0.339%."
    ),
    "script": "scratch/pitch_checks/2026-09-08/c1b_september_subcell.py",
    "source": "stand_down",
    "expires": "2027-09-08",
    "note": "",
})

# --- 4. Prune anything past its expiry.
still_live = []
for e in entries:
    exp = str(e.get("expires", "")).strip()
    if exp and exp < ASOF:
        e["note"] = (e.get("note", "") + f" [EXPIRED {exp}, pruned {ASOF}]").strip()
        expired.append(e)
    else:
        still_live.append(e)

w["entries"] = still_live
w["expired"] = expired
w["asof"] = ASOF
w["generated"] = ASOF
P.write_text(json.dumps(w, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"watchlist: {len(still_live)} active, {len(expired)} expired")
for e in still_live[-2:]:
    print("  + added:", e["title"])
print("  - retired:", ita["title"])
