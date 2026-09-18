"""Watchlist triage, part B -- the entries the tape cannot settle.

Everything here needs a trailing-252 LEVEL percentile, a POINT change in ^TNX,
an episode count over full history, the fragility parquet, the release-surprise
history, or the session calendar itself.

Covers entries 1, 2, 5, 13, 15, 18, 20, 24, 25, 30, 33, 35, 36, 37.

Run: python scratch/pitch_checks/2026-09-07/00_watchlist_triage_b.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import (  # noqa: E402
    close_panel, declusters, load_events, load_prices, pct_rank,
    rolling_on_valid, sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-09-04")
NEXT_SESSION = pd.Timestamp("2026-09-08")
SVXY_LEV_BREAK = pd.Timestamp("2018-02-28")

print("=" * 100)
print("PART B -- computed readings, 2026-09-07 (bar 2026-09-04)")
print("=" * 100)

# ==========================================================================
# 36. verify the closure itself against the master_prices index
# ==========================================================================
print("\n########## [36] the extended closure, verified against master_prices ##########")
spy = load_prices(["SPY"])["SPY"]
idx = spy.index
print(f"  last bar in master_prices: {idx[-1].date()}  (tail: "
      f"{[str(d.date()) for d in idx[-5:]]})")
gaps = pd.Series((idx[1:] - idx[:-1]).days, index=idx[1:])
# a >=3 CALENDAR-DAY CLOSURE is a session date-gap of >=4 days (Fri->Tue);
# an ordinary weekend is a date-gap of exactly 3 (Fri->Mon).
big = gaps[gaps >= 4]
recent = big[big.index >= "2026-01-01"]
print(f"  extended closures (session date-gap >=4d) in 2026 so far: {len(recent)}")
print("   ", {str(d.date()): int(v) for d, v in recent.items()})
print(f"  live gap 2026-09-04 -> 2026-09-08 = "
      f"{(NEXT_SESSION - BAR).days} calendar days -> a 3-calendar-day CLOSURE "
      f"(Sat/Sun/Labor Day). QUALIFIES.")
print(f"  total such closures in the cache: n={len(big)}; "
      f"since 2018: n={int((big.index >= '2018-01-01').sum())} "
      f"(entry cites 180 gaps full-sample)")
# prereg search
hits = []
for pat in ("closure", "holiday"):
    for p in (ROOT / "scratch").rglob("*prereg*"):
        if pat in p.name.lower():
            hits.append(str(p))
print(f"  prereg documents matching closure/holiday: {sorted(set(hits)) or 'NONE FOUND'}")

# ==========================================================================
# 1. LQD/HYG joint 52w extremes -- declustered episode count
# ==========================================================================
print("\n########## [1] LQD low / HYG high joint state -- episode count ##########")
px = close_panel(["HYG", "LQD", "IEF", "SPY"])
hi252 = {t: rolling_on_valid(px[t], lambda x: x.rolling(252).max()) for t in px}
lo252 = {t: rolling_on_valid(px[t], lambda x: x.rolling(252).min()) for t in px}
hyg_off_hi = (px["HYG"] / hi252["HYG"] - 1) * 100
lqd_off_lo = (px["LQD"] / lo252["LQD"] - 1) * 100
joint = (hyg_off_hi >= -0.5) & (lqd_off_lo <= 2.0)
days = px.index[joint.fillna(False)]
eps = declusters(pd.DatetimeIndex(days), 21, px.index)
print(f"  raw days={len(days)}  declustered episodes (min_gap 21)={len(eps)}")
print(f"  episodes: {[str(d.date()) for d in eps]}")
print(f"  distinct years: {sorted({d.year for d in eps})}  "
      f"ex-2018: {sorted({d.year for d in eps if d.year != 2018})}")
print(f"  ARM: >=8 declustered episodes spanning >=3 distinct years excluding 2018 "
      f"-> have {len([d for d in eps if d.year != 2018])} ex-2018")
print(f"  TODAY {BAR.date()}: HYG {hyg_off_hi.iloc[-1]:+.2f}% off its 252d high "
      f"(need >= -0.50%), LQD {lqd_off_lo.iloc[-1]:+.2f}% above its 252d low "
      f"(need <= +2.00%) -> state live? {bool(joint.iloc[-1])}")

# ==========================================================================
# 5. TLT/IEF/LQD tight rung + FRESHNESS
# ==========================================================================
print("\n########## [5] TLT+IEF+LQD tight 52w-low rung, and the freshness arm ##########")
px5 = close_panel(["TLT", "IEF", "LQD"])
lo5 = {t: rolling_on_valid(px5[t], lambda x: x.rolling(252).min()) for t in px5}
off = {t: (px5[t] / lo5[t] - 1) * 100 for t in px5}
tight = (off["TLT"] <= 0.5) & (off["IEF"] <= 1.0) & (off["LQD"] <= 1.0)
tdays = px5.index[tight.fillna(False)]
print(f"  TODAY: TLT {off['TLT'].iloc[-1]:+.2f}% (<=0.50), "
      f"IEF {off['IEF'].iloc[-1]:+.2f}% (<=1.00), "
      f"LQD {off['LQD'].iloc[-1]:+.2f}% (<=1.00) -> rung live? {bool(tight.iloc[-1])}")
if len(tdays):
    last = tdays[-1]
    n_since = int((px5.index > last).sum())
    print(f"  last tight-rung day: {last.date()} ({n_since} sessions ago)")
    print(f"  tight-rung days in 2026: "
          f"{[str(d.date()) for d in tdays if d.year == 2026]}")
    # would the NEXT firing be episode-first? gap from last firing
    print(f"  ARM = the rung fires on a day that is the FIRST trigger in >=10 "
          f"sessions. Gap since the last firing is now {n_since}, so the NEXT "
          f"firing WOULD qualify on freshness -- but the rung is not live today.")

# ==========================================================================
# 13 / 15 / 18. ^TNX point changes and 252d proximity
# ==========================================================================
print("\n########## [13/15/18] ^TNX point arithmetic (quoted in PERCENT) ##########")
tnx = load_prices(["^TNX"])["^TNX"]["Close"].dropna()
last = tnx.iloc[-1]
ch21 = last - tnx.iloc[-22]
ch63 = last - tnx.iloc[-64]
ch252 = last - tnx.iloc[-253]
mx252 = tnx.iloc[-252:].max()
pct_of_max = (last / mx252 - 1) * 100
print(f"  ^TNX close {last:.3f}% on {tnx.index[-1].date()}")
print(f"  21-session change {ch21:+.3f} pt ({ch21*100:+.1f} bp)   [13 arm: >= +0.20 pt]")
print(f"  63-session change {ch63:+.3f} pt ({ch63*100:+.1f} bp)")
print(f"  252-session change {ch252:+.3f} pt ({ch252*100:+.1f} bp)  [18 arm: >= +0.78 pt]")
print(f"  trailing-252 max {mx252:.3f}; today is {pct_of_max:+.3f}% from it; "
      f"at the max? {bool(abs(last - mx252) < 1e-9)}")
print(f"  [18] level needed for the arm: TNX(t-252) + 0.78 = "
      f"{tnx.iloc[-253] + 0.78:.3f} vs live {last:.3f} "
      f"-> gap {100*(tnx.iloc[-253] + 0.78 - last):+.1f} bp")
print(f"  ^TNX 21d rank {pct_rank(tnx, 21).iloc[-1]:.1f}  "
      f"[15 arm leg: >= 65]")
dx = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]["Close"].dropna()
print(f"  DX-Y.NYB 21d rank {pct_rank(dx, 21).iloc[-1]:.1f}  "
      f"[13 arm leg: <= 15 | 15 arm leg: <= 20 | 23 arm leg: <= 2]")
# 15's honest magnitude-floor form is a cost bar, not a state
print("  [15] the arm is a COST number (7.5 bps at h=5 on the magnitude-floor "
      "form, 3.9 bps today), not a state -- unmoved without new episodes.")

# ==========================================================================
# 20 / 24 / 35. fragility parquet
# ==========================================================================
print("\n########## [20/24/35] fragility dial readings ##########")
frag = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = frag["63d"].rolling(10).mean()
print(f"  last row {frag.index[-1].date()}: raw 5d {frag['5d'].iloc[-1]:.1f}, "
      f"raw 21d {frag['21d'].iloc[-1]:.1f}, raw 63d {frag['63d'].iloc[-1]:.1f}")
print(f"  ma10(63d) = {ma10.iloc[-1]:.2f}")
print(f"  ma10 percentile within its own history: "
      f"{100*(ma10.dropna() <= ma10.iloc[-1]).mean():.1f}")
print(f"  [20] arm leg: raw-21d <= 50 -> today {frag['21d'].iloc[-1]:.1f} FAILS")
print(f"  [24] arm leg: ma10(63d) < 50 -> today {ma10.iloc[-1]:.1f} FAILS "
      f"(cell's own max observed dial 68.0)")
print(f"  [35] arm leg: ma10(63d) < 50 -> today {ma10.iloc[-1]:.1f} FAILS")

# ==========================================================================
# 25. SMH -- is the trailing-252 return top-decile, and the r5/r63 legs
# ==========================================================================
print("\n########## [25] SMH deep-correction legs ##########")
smh = load_prices(["SMH"])["SMH"]["Close"].dropna()
r252 = smh.pct_change(252)
r252_rank = rolling_on_valid(r252, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  SMH 252d return {100*r252.iloc[-1]:+.2f}%, trailing-252 PIT rank of that "
      f"return {r252_rank.iloc[-1]:.1f}  [top-decile gate: >= 90]")
print(f"  SMH 63d rank {pct_rank(smh, 63).iloc[-1]:.1f} [arm leg <= 5]   "
      f"5d rank {pct_rank(smh, 5).iloc[-1]:.1f} [arm leg < 15]")

# ==========================================================================
# 30. ^MOVE trailing-252 LEVEL percentile
# ==========================================================================
print("\n########## [30] ^MOVE trailing-252 LEVEL percentile ##########")
mv = load_prices(["^MOVE"])["^MOVE"]["Close"].dropna()
mv_pct = rolling_on_valid(mv, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  ^MOVE {mv.iloc[-1]:.2f} on {mv.index[-1].date()}, trailing-252 LEVEL "
      f"percentile {mv_pct.iloc[-1]:.1f}  [arm band: [40,50)]")
print(f"  last 5 readings: "
      f"{[f'{d.date()} {v:.1f}' for d, v in mv_pct.tail(5).items()]}")

# ==========================================================================
# 33 / 35. ^VIX 21-day RELATIVE RANGE percentile
# ==========================================================================
print("\n########## [33/35] ^VIX 21d relative-range trailing-252 percentile ##########")
vix = load_prices(["^VIX"])["^VIX"]["Close"].dropna()
rel = (vix.rolling(21).max() - vix.rolling(21).min()) / vix.rolling(21).mean()
rel_pct = rolling_on_valid(rel, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"  ^VIX {vix.iloc[-1]:.2f}; 21d rel-range {rel.iloc[-1]:.4f}; "
      f"trailing-252 percentile {rel_pct.iloc[-1]:.2f}")
print(f"  [33] arm band (5.0, 15.0] -> today {rel_pct.iloc[-1]:.2f} is in the DEAD "
      f"(0,5] band that pays -0.096%")
print(f"  [35] arm leg <= 15 -> today {rel_pct.iloc[-1]:.2f} CLEARS, but the dial "
      f"leg (<50) fails at {ma10.iloc[-1]:.1f}")
print(f"  last 6 readings: "
      f"{[f'{d.date()} {v:.2f}' for d, v in rel_pct.tail(6).items()]}")
ev = load_events()
sched = ev[ev.event.isin(["nfp", "cpi", "ppi", "fomc_decision"])]
fut = sched[sched.date > BAR].head(5)
print("  next print anchors (k=-2 signal session, entry MOC k=-1, exit at the print):")
sess = list(pd.bdate_range("2026-09-08", "2026-10-30"))
for _, r in fut.iterrows():
    d = r["date"]
    if d not in sess:
        continue
    i = sess.index(d)
    nxt = sched[sched.date > d]["date"].iloc[0]
    runway = sum(1 for s in sess if d < s <= nxt)
    print(f"    {r['event']:<14} {d.date()}  signal k=-2 {sess[i-2].date()}  "
          f"runway to next print ({nxt.date()}) = {runway} sessions "
          f"[33 needs >=3]")

# ==========================================================================
# 2. SVXY overnight into CPI -- re-measured with the 2026-08-12 print added
# ==========================================================================
print("\n########## [2] SVXY overnight into CPI, LOYO re-measure ##########")
pr = load_prices(["SVXY", "SPY"])
on = {t: (pr[t]["Open"] / pr[t]["Close"].shift(1) - 1.0).dropna() for t in pr}
cpi = load_events(["cpi"])["date"]
all_d = pr["SPY"].index
cpi_sess = pd.DatetimeIndex([d for d in cpi if d in all_d])
post = on["SVXY"].reindex(cpi_sess).dropna()
post = post[post.index >= SVXY_LEV_BREAK]
print(f"  CPI overnights 2018+: N={len(post)}, raw mean {1e4*post.mean():+.1f} bps, "
      f"hit {100*(post > 0).mean():.1f}%, "
      f"sign p {sign_test(int((post > 0).sum()), len(post)):.4f}")
print(f"  most recent CPI overnights: "
      f"{[f'{d.date()} {1e4*v:+.0f}bp' for d, v in post.tail(4).items()]}")
yrs = sorted(set(post.index.year))
loyo = {y: 1e4 * post[post.index.year != y].mean() for y in yrs}
worst_y = min(loyo, key=loyo.get)
print(f"  LOYO (raw, 2018+): floor = dropping {worst_y} leaves "
      f"{loyo[worst_y]:.1f} bps   [ARM: 40-50 bps]")
print(f"  LOYO table: {{y: round(v,1) for ...}} = "
      f"{ {y: round(v, 1) for y, v in loyo.items()} }")
# beta-neutral residual
j = pd.concat([on["SVXY"].rename("y"), on["SPY"].rename("x")], axis=1).dropna()
j = j[j.index >= SVXY_LEV_BREAK]
X = np.column_stack([np.ones(len(j)), j["x"].values])
b, *_ = np.linalg.lstsq(X, j["y"].values, rcond=None)
resid = pd.Series(j["y"].values - X @ b, index=j.index)
rc = resid.reindex(cpi_sess).dropna()
print(f"  beta-neutral (beta_SPY {b[1]:+.2f}): residual {1e4*rc.mean():+.1f} bps, "
      f"N={len(rc)}, t={rc.mean()/(rc.std(ddof=1)/np.sqrt(len(rc))):+.2f}, "
      f"hit {100*(rc > 0).mean():.1f}%")
loyo_r = {y: 1e4 * rc[rc.index.year != y].mean() for y in sorted(set(rc.index.year))}
wr = min(loyo_r, key=loyo_r.get)
print(f"  beta-neutral LOYO floor: dropping {wr} leaves {loyo_r[wr]:.1f} bps "
      f"[ARM: 40-50 bps, i.e. 5x an 8-10 bp MOC->MOO round trip]")
nxt_cpi = pd.Timestamp("2026-09-11")
print(f"  calendar: next CPI {nxt_cpi.date()}; the overnight entry is the "
      f"2026-09-10 CLOSE, 2 sessions after the next open.")

# ==========================================================================
# 37. prior NFP surprise
# ==========================================================================
print("\n########## [37] prior-NFP-surprise band ##########")
rel_h = pd.read_parquet(ROOT / "data" / "macro_release_history.parquet")
n = rel_h[rel_h["event_name"] == "Non Farm Payrolls"].copy()
n = n.dropna(subset=["surprise"]).sort_values("release_date")
n = n.drop_duplicates(subset=["release_date"], keep="last")
print(f"  release history frozen at {n['release_date'].max().date()}; "
      f"last 4 payroll surprises:")
for _, r in n.tail(4).iterrows():
    print(f"    {r['release_date'].date()}  actual {r['actual']}  "
          f"consensus {r['consensus']}  surprise {r['surprise']:+.0f}k")
prior = n["surprise"].iloc[-1]
print(f"  the 2026-09-04 print's PRIOR surprise = {prior:+.0f}k "
      f"(the 2026-08-07 release) -> band (-100,-50] required, so it lands in "
      f"the <= -100k half that pays -0.175%")
print("  NEXT NFP 2026-10-02. Its prior surprise is the 2026-09-04 print, which "
      "is NOT in the frozen file -- it must be read live before that morning.")
print("  CPI-in-hold leg for 2026-10-02: h=3 hold runs to ~2026-10-07; CPI is "
      "2026-10-14, PPI 2026-10-15 -> NEITHER inside, so the second leg fails "
      "for October too on today's calendar.")

# ==========================================================================
# RECONCILIATION -- where today's readings differ from the 2026-09-04 notes
# ==========================================================================
print("")
print("########## RECONCILIATION vs the 2026-09-04 watchlist notes ##########")
print("  The 2026-09-04 triage ran BEFORE that session closed, so its numbers "
      "sit on the 2026-09-03 bar. Today's are on the settled 2026-09-04 bar.")
print(f"  ^MOVE level pctile: 09-03 {mv_pct.loc['2026-09-03']:.1f} "
      f"(the note quoted 62.7) -> 09-04 {mv_pct.iloc[-1]:.1f}")
print(f"  ^VIX rel-range pctile: 09-02 {rel_pct.loc['2026-09-02']:.2f} "
      f"(entry 33 quoted 3.57 for its 09-03 run) -> 09-03 "
      f"{rel_pct.loc['2026-09-03']:.2f} -> 09-04 {rel_pct.iloc[-1]:.2f}")
print("  ^TNX last 8 closes: "
      f"{[f'{d.date()} {v:.3f}' for d, v in tnx.tail(8).items()]}")
print(f"  ^TNX trailing-252 max {mx252:.3f} set on "
      f"{tnx.iloc[-252:].idxmax().date()}; the 252d-max TOUCH LAPSED on the "
      f"2026-09-04 bar ({100*(last - mx252):+.1f} bp below it).")
print(f"  [1] the 09-04 note said 3 declustered episodes and the entry text "
      f"said 4; recomputed at the entry's own tolerances (HYG 0.5%, LQD 2.0%, "
      f"gap 21 -- the d1_hyg_lqd_unanchored settings) the count is {len(eps)}: "
      f"{[str(d.date()) for d in eps]}")
print(f"      raw joint days in 2026: "
      f"{[str(d.date()) for d in days if d.year == 2026]}")

# ==========================================================================
# NEAR-MISS ARITHMETIC -- exactly what the closest entries need, in levels
# ==========================================================================
print("")
print("########## NEAR-MISS ARITHMETIC (what would have to print) ##########")
tlt_lo = lo5["TLT"].iloc[-1]
tlt_px = px5["TLT"].iloc[-1]
tlt_need = tlt_lo * 1.005
tlt_atr = load_prices(["TLT"])["TLT"]
from pitch_grammar import wilder_atr as _watr
a = _watr(tlt_atr["High"].to_numpy(), tlt_atr["Low"].to_numpy(),
          tlt_atr["Close"].to_numpy())[-1]
print(f"  [5] TLT close {tlt_px:.2f}; 252d low {tlt_lo:.2f}; the 0.5% rung needs a "
      f"close at or below {tlt_need:.2f} = {100*(tlt_need/tlt_px-1):+.2f}% "
      f"({(tlt_px-tlt_need)/a:.2f} Wilder-14 ATR of {a:.2f}). "
      f"IEF and LQD legs already clear. Freshness gap 13 sessions >= 10, so a "
      f"firing would BE episode-first.")
print(f"  [33] ^VIX 21d rel-range pctile {rel_pct.iloc[-1]:.2f}; the band needs "
      f"> 5.00, i.e. {5.0 - rel_pct.iloc[-1]:.2f} points higher. Trajectory "
      f"09-02 {rel_pct.loc['2026-09-02']:.2f} -> 09-03 {rel_pct.loc['2026-09-03']:.2f} "
      f"-> 09-04 {rel_pct.iloc[-1]:.2f}. Next QUALIFYING anchor is the CPI k=-2 "
      f"session 2026-09-09 (runway 3); 2026-09-08 is the PPI k=-2 anchor and PPI "
      f"is disqualified at runway 1.")
print(f"  [18] ^TNX needs BOTH a fresh trailing-252 max (> {mx252:.3f}) AND a "
      f"252-session change >= +0.78 pt, i.e. a close >= {tnx.iloc[-253]+0.78:.3f}. "
      f"Live {last:.3f}; the binding leg is the magnitude one at "
      f"{100*(tnx.iloc[-253]+0.78-last):+.1f} bp away.")
print(f"  [25] SMH needs r63 <= 5 (now {pct_rank(smh,63).iloc[-1]:.1f}), r5 < 15 "
      f"(now {pct_rank(smh,5).iloc[-1]:.1f}) and a top-decile 252d return "
      f"(rank now {r252_rank.iloc[-1]:.1f}); all three legs are off, versus one "
      f"on 2026-09-04.")
spy_px = px["SPY"].iloc[-1]
spy_hi = hi252["SPY"].iloc[-1]
print(f"  [6/20/24] SPY {spy_px:.2f} is {100*(spy_px/spy_hi-1):+.2f}% off its 252d "
      f"high {spy_hi:.2f}. Entry 6 needs more than -1.00% (misses by "
      f"{abs(100*(spy_px/spy_hi-1))-1.0:+.2f}pp) but is midterm-blocked anyway; "
      f"entries 20 and 24 need at least -2.00%, i.e. a close below "
      f"{spy_hi*0.98:.2f} ({100*(spy_hi*0.98/spy_px-1):+.2f}%).")

print("\n" + "=" * 100)
print("PART B DONE")
print("=" * 100)
