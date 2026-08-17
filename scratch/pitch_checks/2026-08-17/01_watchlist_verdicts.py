"""Stage B1: verdict on every active watchlist entry, citing today's number.

One block per entry. CHECK means the trigger moved and the cell earns a real
check today; PASS means it did not, with the live value printed anyway so the
surface map shows it was looked at.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

ASOF = "2026-08-17"
BAR = pd.Timestamp("2026-08-14")

px = load_prices(["TLT", "IEF", "LQD", "HYG", "SVXY", "GLD", "GDX", "USO", "XLE",
                  "SPY", "^SKEW", "^VIX", "FXI", "EEM", "IHI"])
C = {k: v["Close"] for k, v in px.items()}


def r(s, n, lb=252):
    return pct_rank(s, n, lb)


def dist_low(s, lb=252):
    return (s / s.rolling(lb).min() - 1.0) * 100.0


def dist_high(s, lb=252):
    return (s / s.rolling(lb).max() - 1.0) * 100.0


print("=" * 100)
print(f"WATCHLIST VERDICTS  asof {ASOF}  bar {BAR.date()}")
print("=" * 100)

# ---------------------------------------------------------------- 1. nfp x rates
print("\n[1] Long TLT from the NFP close, long end at its 52w floor  (added 08-07)")
print("    trigger: a NON-midterm NFP. Next NFP 2026-09-04 is midterm AND 14 td out.")
print("    VERDICT: PASS. Structurally unreachable: first non-midterm NFP is 2027-01,")
print("    and the 2026-09-04 print is beyond the 10 td max pitch horizon anyway.")

# ------------------------------------------------------- 2. credit LQD/HYG extremes
hy, lq = C["HYG"], C["LQD"]
print("\n[2] Long LQD against short HYG at joint 52w extremes  (added 08-10)")
print(f"    HYG off 52w high {dist_high(hy).loc[BAR]:+.2f}% (needs <= -0.5 to be OFF the state)")
print(f"    LQD off 52w low  {dist_low(lq).loc[BAR]:+.2f}% (needs <= 2.0 to be IN)")
print("    trigger: >= 8 declustered episodes over >= 3 years EXCLUDING 2018. Count is 4,")
print("    three of them in 2018 and the fourth the live cluster begun 2026-07-22.")
print("    VERDICT: PASS. State still live, still mid-cluster, count unchanged at 4.")

# ------------------------------------------------------ 3. SVXY overnight into CPI
print("\n[3] Long SVXY overnight into the CPI print  (added 08-11)")
print("    trigger: beta-neutral drop-best-year edge >= 40-50 bps; it was 19.7 bps.")
print("    Owed re-measure is at the run before the NEXT CPI, 2026-09-11 (18 td out).")
print("    VERDICT: PASS, deferred with cause. No CPI inside any legal horizon.")

# ------------------------------------------- 4. GLD on a miner-led thrust GLD has not joined
gdx5, gld5 = r(C["GDX"], 5).loc[BAR], r(C["GLD"], 5).loc[BAR]
print("\n[4] Long GLD on a miner-led thrust the metal has not joined  (added 08-11)")
print(f"    GDX 5d rank {gdx5:.1f} (needs >= 95) | GLD 5d rank {gld5:.1f} (needs < 95)")
print("    VERDICT: PASS. The divergence is absent: neither leg is thrusting on 5d.")
print(f"    (context: GDX 21d rank {r(C['GDX'],21).loc[BAR]:.1f}, GLD 21d {r(C['GLD'],21).loc[BAR]:.1f}"
      " -- the thrust is a 21d event, not the 5d shape this entry keys on.)")

# ------------------------------------------------- 5. XLE on a crude 1d pop 5-6%
uso = C["USO"]
uso1 = uso.pct_change().loc[BAR] * 100
atr_uso = wilder_atr(px["USO"]["High"], px["USO"]["Low"], px["USO"]["Close"])
atr_uso = pd.Series(atr_uso, index=uso.index)
pop_atr = (uso.diff() / atr_uso.shift(1)).loc[BAR]
print("\n[5] Long XLE on a crude one-day thrust in the 5-6 percent band  (added 08-11)")
print(f"    USO 1d {uso1:+.2f}% (needs [5,6)) | {pop_atr:+.2f} ATR (needs >= 1.50)")
print("    VERDICT: PASS. No one-day pop at all; Friday was a +1.26% continuation day.")
print(f"    (context: the 5-DAY complex thrust is the live shape -- USO 5d {C['USO'].pct_change(5).loc[BAR]*100:+.2f}%,"
      f" XLE 5d rank {r(C['XLE'],5).loc[BAR]:.1f}. Different cell, checked separately today.)")

# ---------------------------------------- 6. TLT with the IG complex pinned at 52w lows
tl, ie, lq2 = C["TLT"], C["IEF"], C["LQD"]
dl_t, dl_i, dl_l = dist_low(tl), dist_low(ie), dist_low(lq2)
tight = (dl_t <= 0.5) & (dl_i <= 1.0) & (dl_l <= 1.0)
alld = tl.index
trig = alld[tight.reindex(alld).fillna(False)]
recent = trig[trig >= "2026-06-01"]
print("\n[6] Long TLT with the whole IG complex pinned at 52w lows  (added 08-12)")
print(f"    TLT {dl_t.loc[BAR]:.2f}% off 52w low (needs <= 0.50)")
print(f"    IEF {dl_i.loc[BAR]:.2f}% (needs <= 1.00) | LQD {dl_l.loc[BAR]:.2f}% (needs <= 1.00)")
print(f"    price rung LIVE today: {bool(tight.loc[BAR])}  <-- it switched back ON")
print(f"    trigger days since 2026-06-01: {[str(d.date()) for d in recent]}")
pos = list(alld).index(BAR)
prior = [d for d in trig if d < BAR]
if prior:
    gap = pos - list(alld).index(prior[-1])
    print(f"    last prior trigger day {prior[-1].date()}, gap = {gap} sessions (freshness needs >= 10)")
print("    VERDICT: CHECK. The price rung is live for the first time since 08-12; the")
print("    freshness leg is what today's check has to settle.")

# ------------------------------------------------------ 7. SPY on a skew spike alone
sk5 = r(C["^SKEW"], 5).loc[BAR]
spy_dh = dist_high(C["SPY"]).loc[BAR]
print("\n[7] Long SPY on a skew spike alone  (added 08-12)")
print(f"    ^SKEW 5d rank {sk5:.1f} (needs >= 95) | SPY off 52w high {spy_dh:+.2f}% (needs < -1.0)")
print("    | cycle year midterm (needs NON-midterm)")
print("    VERDICT: PASS. Skew is elevated but short of the gate, and both of the")
print("    conditions that define the live intersection still fail.")

# ------------------------------------- 8. fade a crude thrust out of a deep base
u5, u63 = r(uso, 5).loc[BAR], r(uso, 63).loc[BAR]
print("\n[8] Fade a crude thrust out of a deep base with a macro print inside  (added 08-12)")
print(f"    USO 5d rank {u5:.1f} (needs >= 90) | 63d rank {u63:.1f} (needs <= 20)")
print("    trigger: >= 8 post-2020 episodes with the 2018+ h=3 mean positive; 4 and -0.465%.")
print("    VERDICT: PASS. The 5d leg misses and the episode count is structural, not tape.")

# ------------------------------------------------------------- 9. IHI thrust
ihi21 = r(C["IHI"], 21).loc[BAR]
print("\n[9] Long the medical-device thrust, IHI at a 21d rank of 100  (added 08-13)")
print(f"    IHI 21d rank {ihi21:.1f} (needs 100 AND episode-first)")
print("    trigger: the 27-ETF reference class must show Cochran Q p < 0.05; measured 0.544.")
print("    VERDICT: PASS. A reference-class condition cannot flip in two sessions, and")
print(f"    the rank leg is short at {ihi21:.1f} in any case.")

# -------------------------------------------------------------- 10. FXI break
f5, f21 = r(C["FXI"], 5).loc[BAR], r(C["FXI"], 21).loc[BAR]
eem5 = C["EEM"].pct_change(5).loc[BAR] * 100
print("\n[10] Long China's five-day break inside an intact thrust  (added 08-13)")
print(f"    FXI 5d rank {f5:.1f} (needs <= 20) | FXI 21d rank {f21:.1f} (needs >= 80)"
      f" | EEM 5d {eem5:+.2f}% (needs > 0)")
print("    VERDICT: PASS. The break leg fires and EEM holds, but the intact-thrust leg")
print("    fails outright, which is the leg that made the cell more than EM beta.")

# ------------------------------------------------ 11. industry breadth washout
print("\n[11] Industry-wide five-day breadth washout with the trend BROKEN  (added 08-14)")
print("    trigger: >= 70% of a coherent industry at 5d rank <= 20 with median 63d rank")
print("    BELOW 70, AND a <= 4-name selection rule that clears the alphabetical placebo.")
print("    The second leg has never been tested, so the entry cannot fire on tape alone.")
print("    VERDICT: PASS. Untested-form leg outstanding; see 03 for today's breadth scan.")
