"""Watchlist triage, part A -- the 39 ACTIVE entries settled from the TAPE and
the CALENDAR (data/pitch_tape.json on the 2026-09-04 bar + data/macro_events.csv).

Today is 2026-09-07, Labor Day, NYSE CLOSED. Freshest bar 2026-09-04, next
session 2026-09-08. Every entry that keys on a rank, a distance-from-extreme, a
one-day move, a cross-sectional count over the 218-name tape, or a calendar date
is settled here. Entries needing a trailing-252 LEVEL percentile, a point change
in ^TNX, an episode count over full history, the fragility parquet, or the
release-surprise history are in 00_watchlist_triage_b.py.

Run: python scratch/pitch_checks/2026-09-07/00_watchlist_triage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import load_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
TODAY = pd.Timestamp("2026-09-07")
BAR = pd.Timestamp("2026-09-04")
NEXT_SESSION = pd.Timestamp("2026-09-08")
MIDTERM = TODAY.year % 4 == 2

tape = json.loads((ROOT / "data" / "pitch_tape.json").read_text(encoding="utf-8"))
T = tape["tickers"]
print(f"tape asof={tape['asof']} freshest_bar={tape['freshest_bar']} "
      f"n={len(T)}   today={TODAY.date()} (NYSE CLOSED)  next={NEXT_SESSION.date()}  "
      f"midterm={MIDTERM}")
stale = sorted(t for t, m in T.items() if m["date"] != tape["freshest_bar"])
print(f"stale tickers vs freshest bar: {stale}")


def g(t: str, k: str):
    m = T.get(t)
    return None if m is None else m.get(k)


def fmt(v, d=2):
    return "n/a" if v is None else f"{v:+.{d}f}"


VERDICTS: list[tuple[int, str, str, str, str]] = []


def verdict(i, title, v, number, why):
    VERDICTS.append((i, title, v, number, why))
    print(f"\n[{i:>2}] {v:<12} {title}")
    print(f"      NUMBER: {number}")
    print(f"      WHY   : {why}")


ev = load_events()
ev2026 = ev[(ev.date >= "2026-09-01") & (ev.date <= "2026-12-31")]
print("\nlive calendar (2026-09-01 .. 2026-12-31):")
print(ev2026[["date", "event"]].to_string(index=False))

print("\n" + "#" * 100)
print("# PART A -- tape- and calendar-settled entries")
print("#" * 100)

# ---------------------------------------------------------------- 0
verdict(
    0, "Long TLT from the NFP close, long end at 52w floor",
    "DATE-PARK",
    f"2026 % 4 == {TODAY.year % 4} (midterm). NFP was 2026-09-04 (last session); "
    f"next NFP 2026-10-02, still midterm. First non-midterm NFP = 2027-01.",
    "cell is midterm-dead (+0.071%, N=12) and alive only outside midterms; "
    "no non-midterm payroll print until 2027-01.")

# ---------------------------------------------------------------- 3
gdx_r5, gld_r5 = g("GDX", "rank_5d"), g("GLD", "rank_5d")
gld_r63, gld_dh = g("GLD", "rank_63d"), g("GLD", "dist_52w_high_pct")
verdict(
    3, "Long GLD on a miner-led thrust the metal has not joined",
    "PASS",
    f"GDX r5 {gdx_r5} vs >=95 | GLD r5 {gld_r5} vs <95 | GLD r63 {gld_r63} vs >=50 | "
    f"GLD dist_52wh {fmt(gld_dh)}% vs within -10%",
    "all four legs fail -- the miner thrust is absent AND the two trend legs added "
    "on 2026-08-21 are both short; PPI 09-10 + CPI 09-11 would also sit inside a 5d "
    "hold entered 09-08, which the original three-leg form already excluded.")

# ---------------------------------------------------------------- 4
uso_1d = g("USO", "ret_1d")
verdict(
    4, "Long XLE on a crude one-day thrust in the [5,6)% band",
    "PASS",
    f"USO 1-day {fmt(uso_1d)}% vs the [5.0,6.0)% band",
    "no band episode, so neither arm number (crude-beta residual sign p <=0.10 at "
    ">65% hit on >=20 episodes; [4,5)% bucket ceasing to be negative) can have moved.")

# ---------------------------------------------------------------- 6
skew_r5, spy_dh = g("^SKEW", "rank_5d"), g("SPY", "dist_52w_high_pct")
verdict(
    6, "Long SPY on a skew spike alone",
    "PASS",
    f"^SKEW r5 {skew_r5} vs >=95 | SPY dist_52wh {fmt(spy_dh)}% vs more than -1% | "
    f"midterm={MIDTERM} vs non-midterm required",
    "all three legs fail; the cycle leg alone parks the entry to 2027 even if the "
    f"skew spike arrives (^SKEW tape date {g('^SKEW','date')}).")

# ---------------------------------------------------------------- 7
uso_r5, uso_r63 = g("USO", "rank_5d"), g("USO", "rank_63d")
verdict(
    7, "Fade a crude thrust out of a deep base, macro print inside the hold",
    "PASS",
    f"USO r5 {uso_r5} vs >=90 | USO r63 {uso_r63} vs <=20",
    "no new episode of the parent state, so the post-2020 episode count is "
    "unchanged at 4 against the >=8 arm.")

# ---------------------------------------------------------------- 8
ihi_r21 = g("IHI", "rank_21d")
verdict(
    8, "Long the medical-device thrust, IHI at a 21d rank of 100",
    "PASS",
    f"IHI r21 {ihi_r21} vs the rank-100 rung",
    "state not live, so the 27-ETF reference class (Cochran Q p 0.544, "
    "family-wise p 0.9330) is unchanged.")

# ---------------------------------------------------------------- 9
fxi_r5, fxi_r21, eem_r5 = g("FXI", "rank_5d"), g("FXI", "rank_21d"), g("EEM", "ret_5d")
verdict(
    9, "Long China's five-day break inside an intact thrust",
    "PASS",
    f"FXI r5 {fxi_r5} vs <=20 | FXI r21 {fxi_r21} vs >=80 | EEM 5d {fmt(eem_r5)}% vs >0",
    "the break leg fails; residual-vs-EEM arm (-0.277% today) needs new episodes.")

# ---------------------------------------------------------------- 10
verdict(
    10, "Long TLT on the NOVEMBER month-position effect",
    "DATE-PARK",
    "entry window = trading days 4-12 of November 2026, i.e. ~2026-11-05 to "
    "2026-11-17. Today 2026-09-07 is not in it.",
    "pure calendar arm; nothing in September can turn it on.")

# ---------------------------------------------------------------- 11
tlt_dl = g("TLT", "dist_52w_low_pct")
verdict(
    11, "Short SPY at a 52w high while the long end sits at a 52w low",
    "PASS",
    f"SPY dist_52wh {fmt(spy_dh)}% vs within 0.5% | TLT dist_52wl {fmt(tlt_dl)}% vs within 1%",
    "both legs fail, so no new instance of the joint state and the de-concentrated "
    "mean (+0.039% per episode, 1.3x cost) is unchanged.")

# ---------------------------------------------------------------- 12
vix_1d, vix_r21, spy_1d = g("^VIX", "ret_1d"), g("^VIX", "rank_21d"), g("SPY", "ret_1d")
verdict(
    12, "Long SPY on a volatility pop inside an already-calm tape",
    "PASS",
    f"^VIX 1-day {fmt(vix_1d)}% vs >=+5% | ^VIX r21 {vix_r21} vs <=25 | "
    f"SPY 1-day {fmt(spy_1d)}% vs down less than 0.75%",
    "the defining pop leg is wrong-signed; no new episode, so the Welch t of the "
    "increment over calm-tape-alone stays +1.09 against the >=2.0 arm.")

# ---------------------------------------------------------------- 14
xlv_1d, xlk_1d = g("XLV", "ret_1d"), g("XLK", "ret_1d")
gap = None if (xlv_1d is None or xlk_1d is None) else xlv_1d - xlk_1d
spy_atr_pct = g("SPY", "atr_pct")
verdict(
    14, "Long tech against healthcare after a rotation gap",
    "PASS",
    f"XLV-XLK 1-day gap {fmt(gap)}pp vs >=+3.0pp | SPY dist_52wh {fmt(spy_dh)}% vs "
    f"within 3% (clears) | SPY ATR {spy_atr_pct}% vs <1.2% (clears)",
    "the rotation leg is wrong-signed, so no new subclass episode toward the "
    "3-new-winners-outside-2026 arm.")

# ---------------------------------------------------------------- 16
tlt_1d = g("TLT", "ret_1d")
verdict(
    16, "Short TLT after a big up day from inside the 52-week low zone",
    "PASS",
    f"TLT 1-day {fmt(tlt_1d)}% vs >=+1.5% | TLT dist_52wl {fmt(tlt_dl)}% vs within 4% (clears)",
    "the thrust leg fails; the arm is the [1.0,1.5)% band ceasing to be "
    "wrong-signed (-0.241% at a 30.8% hit), which no non-episode session moves.")

# ---------------------------------------------------------------- 17
BANKS = ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "KEY", "RF", "STT", "SCHW"]
b_r5 = {t: g(t, "rank_5d") for t in BANKS}
b_r63 = {t: g(t, "rank_63d") for t in BANKS}
n_wash = sum(1 for v in b_r5.values() if v is not None and v <= 20)
med63 = pd.Series([v for v in b_r63.values() if v is not None]).median()
verdict(
    17, "Short KRE against XLF on a bank breadth washout",
    "PASS",
    f"{n_wash} of 11 bank names at r5<=20 ({100*n_wash/11:.1f}%) vs the >=70% rung; "
    f"complex median r63 {med63:.1f}",
    "no washout, and the arm is an ex-crisis cost threshold (+0.102% at h=3 vs the "
    "+0.35% required) that only new modern episodes move.")
print("      per-name r5:", {t: b_r5[t] for t in BANKS})

# ---------------------------------------------------------------- 19
ENERGY = ["XLE", "XOP", "USO", "COP", "CVX", "VLO", "OXY", "SLB", "EOG", "HAL", "WMB"]
e_z = {t: g(t, "z10") for t in ENERGY}
n_thrust = sum(1 for v in e_z.values() if v is not None and v >= 2.0)
mx = max((v for v in e_z.values() if v is not None), default=None)
mx_t = max(((v, t) for t, v in e_z.items() if v is not None), default=(None, None))[1]
verdict(
    19, "The NARROW energy thrust cluster, 2 or 3 names at z10 >= 2",
    "PASS",
    f"count at z10>=2.0 is {n_thrust} vs the [2,3] arm; max z10 {fmt(mx)} ({mx_t})",
    "the count is on the zero side of the arm, not the five side; nothing to trade.")
print("      per-name z10:", e_z)

# ---------------------------------------------------------------- 21
SPDRS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
w21 = {t: (g(t, "rank_5d"), g(t, "dist_52w_high_pct")) for t in SPDRS}
hold21 = [t for t, (r5, dh) in w21.items()
          if r5 is not None and dh is not None and r5 <= 5 and dh >= -5.0]
verdict(
    21, "Sector washout into a 52-week high at h=7, as a FAMILY effect",
    "PASS",
    f"sectors holding r5<=5 AND within 5% of the 52w high: {hold21 or 'none'} "
    f"(XLI r5 {g('XLI','rank_5d')}, dist_52wh {fmt(g('XLI','dist_52w_high_pct'))}%)",
    "no family instance, so Cochran Q p 0.789 / XLI 2-of-9 by |t| is unchanged "
    "against the Q p<0.10 + XLI-first arm.")
print("      per-sector (r5, dist_52wh):", w21)

# ---------------------------------------------------------------- 22
xlu_r21, tlt_r21 = g("XLU", "rank_21d"), g("TLT", "rank_21d")
verdict(
    22, "Utilities washout with the long end hit ALONGSIDE it",
    "PASS",
    f"XLU r21 {xlu_r21} vs <=5 | TLT r21 {tlt_r21} vs <25",
    "both legs fail; the rates leg has now been on the wrong side for an eighth "
    "straight session.")

# ---------------------------------------------------------------- 23
dx_r21 = g("DX-Y.NYB", "rank_21d")
verdict(
    23, "The bare dollar washout, long the dollar with no rate leg",
    "DATE-PARK",
    f"2026 % 4 == {TODAY.year % 4} (midterm, wrong-signed at -0.479%); first "
    f"eligible trigger is in 2027. DX r21 {dx_r21} vs <=2 in any case.",
    "cycle arm; the state is not even close, so nothing is being given up.")

# ---------------------------------------------------------------- 24  (dial leg in part B)
hyg_dh = g("HYG", "dist_52w_high_pct")
verdict(
    24, "High yield printing a fresh 52-week high while the index has not",
    "PASS",
    f"HYG dist_52wh {fmt(hyg_dh)}% vs within 0.05% | SPY dist_52wh {fmt(spy_dh)}% vs "
    f"at least -2.0% | dial ma10(63d) -- see part B, arm is <50",
    "all three legs fail; the credit-touch leg is the widest miss.")

# ---------------------------------------------------------------- 25  (252d decile in part B)
smh_r63, smh_r5 = g("SMH", "rank_63d"), g("SMH", "rank_5d")
verdict(
    25, "The leader's deep correction, SMH at a 63-day rank floor",
    "PASS",
    f"SMH r63 {smh_r63} vs <=5 | SMH r5 {smh_r5} vs <15 (the still-falling arm) | "
    f"SMH 252d return {fmt(g('SMH','ret_252d'))}% -- top-decile check in part B",
    "r5 is the binding leg again; the second arm (23-ETF Cochran Q p<0.10 with SMH "
    "first by |t|, today p 0.961 / family-wise 0.8805) needs new episodes regardless.")

# ---------------------------------------------------------------- 26
ief_dl, lqd_dl = g("IEF", "dist_52w_low_pct"), g("LQD", "dist_52w_low_pct")
verdict(
    26, "A pure rates repricing with zero credit stress",
    "PASS",
    f"IEF dist_52wl {fmt(ief_dl)}% vs within 1.5% | LQD dist_52wl {fmt(lqd_dl)}% vs "
    f"within 1.5% | HYG dist_52wh {fmt(hyg_dh)}% vs within 0.25%",
    "the HYG-high leg fails, so the tight rung stays at ONE declustered episode "
    "(2026-08-03) against the >=8 ex-2018/ex-2026 arm.")

# ---------------------------------------------------------------- 27
verdict(
    27, "Long IEF one session out of the Jackson Hole close",
    "DATE-PARK",
    f"2026 % 4 == {TODAY.year % 4} (midterm: +0.037% at t 0.41, 3.7bps vs a 3bps "
    f"round trip). Jackson Hole 2026 was 2026-08-28, seven sessions ago. Next "
    f"eligible anchor 2027-08.",
    "cycle arm plus a spent anchor; nothing this year.")

# ---------------------------------------------------------------- 28
FAM29 = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "EWZ",
         "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
         "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XLE", "XOP", "OIH"]
joint = {t: (g(t, "rank_21d"), g(t, "rank_63d"), g(t, "rank_5d")) for t in FAM29}
holders = [t for t, (r21, r63, r5) in joint.items()
           if None not in (r21, r63) and r21 >= 90 and r63 <= 10]
fired = [t for t in holders if joint[t][2] is not None and joint[t][2] < 15]
verdict(
    28, "The laggard that is STILL FALLING, pooled over 29 ETFs",
    "PASS",
    f"holders of r21>=90 AND r63<=10 across the 29-ETF family: {holders or 'NONE'}; "
    f"of those with r5<15: {fired or 'none'}",
    "with no holder of the joint state there is no name that can print the "
    "sub-15 five-day rank the arm asks for.")

# ---------------------------------------------------------------- 29
slv_1d = g("SLV", "ret_1d")
verdict(
    29, "Short silver after the whole metals complex breaks together",
    "PASS",
    f"SLV 1-day {fmt(slv_1d)}% vs the -4.00%-or-worse depth arm "
    f"(GDX {fmt(g('GDX','ret_1d'))}%, GLD {fmt(g('GLD','ret_1d'))}%)",
    "no break, so the live-depth bucket stays 35-36 against the 46-36 arm; the "
    "lag-profile debt (lag=0 +0.039%, lag=1 +0.516%, lag=2 +0.035%) is untouched.")

# ---------------------------------------------------------------- 31
verdict(
    31, "The small-cap month-end OVERNIGHT in December",
    "DATE-PARK",
    "arm is a December ME-0 in a NON-midterm year, i.e. 2027-12-31 at the "
    "earliest (Dec 2026 is midterm). Today is 2026-09-07.",
    "date arm plus an unpaid max-of-12 month permutation (P 0.476 for IWM).")

# ---------------------------------------------------------------- 32
xle_dh = g("XLE", "dist_52w_high_pct")
verdict(
    32, "Energy at a fresh 52-week high on a session the INDEX fell, h=21",
    "PASS",
    f"XLE dist_52wh {fmt(xle_dh)}% vs a 5bp exact touch | SPY 1-day {fmt(spy_1d)}% "
    f"vs a DOWN session",
    "the at-a-high leg has lapsed and the index leg is wrong-signed; both arm "
    "numbers (h=21 family P, ex-2022/2026 excess) need a fresh episode.")

# ---------------------------------------------------------------- 34
trip = {t: (g(t, "rank_5d"), g(t, "rank_21d"), g(t, "rank_63d")) for t in SPDRS}
floor10 = [t for t, v in trip.items()
           if None not in v and v[0] <= 10 and v[1] <= 10 and v[2] <= 10]
verdict(
    34, "The pooled sector triple rank floor, near-high gate OFF",
    "PASS",
    f"SPDRs at the simultaneous 5/21/63-day rank floor (k<=10): {floor10 or 'NONE'} "
    f"(of 9)",
    "no pooled instance; and the arm is not a threshold anyway -- it needs the "
    "near-high gate's h=10 attribution to turn POSITIVE (today -0.733pp) or a "
    "conditioner the scanner does not already trade (<20 book signals vs 153).")
print("      per-sector (r5, r21, r63):", trip)

print("\n" + "#" * 100)
print("# calendar-only entries around today's closure")
print("#" * 100)

sched = ev[ev.event.isin(["nfp", "cpi", "ppi", "fomc_decision"])]
future = sched[sched.date > BAR].head(6)
print("next scheduled prints after the 2026-09-04 bar:")
print(future[["date", "event"]].to_string(index=False))

# ---------------------------------------------------------------- 36
verdict(
    36, "Risk premium carried ACROSS an extended market closure",
    "UNCOMPUTABLE",
    "the closure IS live (2026-09-04 close -> 2026-09-08 open = 4 calendar days, "
    ">=3), but the arm is a PRE-REGISTERED forward test and NO prereg document "
    "exists in scratch/ultracode_research/ or scratch/pitch_checks/ naming "
    "vehicle/side/entry/exit for this rule.",
    "what is needed: a prereg filed BEFORE a closure, then TWO new >=3-calendar-day "
    "closures observed under it. Today's gap cannot count (nothing was registered "
    "before it). Next countable boundaries: 2026-11-26 Thanksgiving and 2026-12-25 "
    "Christmas -- so the earliest possible arm date is 2026-12-28, and only if a "
    "prereg lands before 2026-11-26.")

# ---------------------------------------------------------------- 38
ppi = pd.Timestamp("2026-09-10")
verdict(
    38, "Long SVXY at the first close AFTER an extended closure",
    "PASS",
    "first close after the closure = 2026-09-08; runway to the next scheduled "
    "print (PPI 2026-09-10) is 2 sessions (09-09, 09-10) vs the >=4 arm.",
    "the entry's own note predicted this exactly; the second, qualitative arm (a "
    "STATED mechanism for why buying vol the session AFTER a closure works when "
    "holding it THROUGH one loses) is also unpaid.")

print("\n" + "=" * 100)
print("PART A SUMMARY")
print("=" * 100)
for i, title, v, number, _ in VERDICTS:
    print(f"{i:>2} | {v:<12} | {title[:58]:<58} | {number[:88]}")
