"""C2 -- ^TNX at a 52-week LEVEL high while ^MOVE sits compressed.

Adversarial round 1. The claim: an ORDERLY grind to a cycle yield high (bond
vol NOT expanding) trends, a disorderly one mean-reverts. Tested BOTH signs on
TLT and IEF, and charged for having looked at both.

Registry constraints honoured here:
- 2026-08-10 MOVE trap: ^MOVE is a LEVEL series. The mechanism needs the LEVEL
  percentile of the trailing year, NOT the 5d return rank (they coincide 30.7%
  of the time). Both are computed and the overlap is printed.
- 2026-08-07: count occurrences BEFORE designing the trade.
- 2026-08-26 depth trap: split the conditioner at TODAY'S LIVE VALUE (44.4),
  not only at the threshold chosen.
- 2026-08-10 tdom/control trap: a rates cell measured against an all-days
  control is invalid -> local +/-126td control is the binding one.
- 2026-08-12: duration proportionality (excess per unit of sd).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

TK = ["^TNX", "^MOVE", "TLT", "IEF", "LQD", "SPY"]
px = close_panel(TK)
print("panel", px.index[0].date(), "..", px.index[-1].date(), len(px), "rows")

# ---------------------------------------------------------------- states
tnx = px["^TNX"]
move = px["^MOVE"]

# ^TNX distance below its trailing-252 LEVEL max (fraction, <=0)
tnx_max = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_dist = tnx / tnx_max - 1.0

# ^MOVE trailing-252 LEVEL percentile (share of last 252 obs <= today)
move_pct = rolling_on_valid(
    move, lambda x: x.rolling(252).apply(
        lambda w: 100.0 * (w <= w[-1]).mean(), raw=True))
# ^MOVE 21-session change (valid sessions)
move_21 = rolling_on_valid(move, lambda x: x / x.shift(21) - 1.0)
# the WRONG statistic, kept only to measure the overlap the registry warns about
move_rank5 = pct_rank(move, 5, 252)

print("\n=== LIVE STATE (asof %s) ===" % px.index[-1].date())
print(f"  ^TNX {tnx.iloc[-1]:.3f}  252d max {tnx_max.iloc[-1]:.3f}  "
      f"dist {100*tnx_dist.iloc[-1]:+.3f}%")
print(f"  ^MOVE {move.iloc[-1]:.2f}  trailing-252 LEVEL pctile "
      f"{move_pct.iloc[-1]:.1f}   21d chg {100*move_21.iloc[-1]:+.2f}%")
print(f"  ^MOVE 5d RETURN rank (the wrong statistic) = {move_rank5.iloc[-1]:.1f}")

both = move_pct.notna() & move_rank5.notna()
lo_lvl = (move_pct <= 50) & both
lo_rnk = (move_rank5 <= 50) & both
print(f"  LEVEL<=50 and RETURN-rank<=50 agree on "
      f"{100*(lo_lvl & lo_rnk).sum()/max(1,int(lo_lvl.sum())):.1f}% of LEVEL days "
      f"(N_level={int(lo_lvl.sum())}, N_rank={int(lo_rnk.sum())})")

# ---------------------------------------------------------------- counts FIRST
print("\n=== STEP 1: COUNT OCCURRENCES BEFORE DESIGNING ANYTHING ===")
usable = tnx_dist.notna() & move_pct.notna()
print(f"  usable rows (both states defined): {int(usable.sum())} "
      f"from {px.index[usable][0].date()}")
for d in (0.0025, 0.005, 0.0075, 0.01, 0.02):
    yh = (tnx_dist >= -d) & usable
    print(f"  TNX within {100*d:.2f}% of 252d high: {int(yh.sum()):5d} days   "
          f"x MOVE<=50pct: {int((yh & (move_pct <= 50)).sum()):5d}   "
          f"x MOVE>50pct: {int((yh & (move_pct > 50)).sum()):5d}")

# pitched rung: within 1.0% of the high (today is -0.527%), MOVE level pctile<=50
YH = 0.01
M_LO = 50.0
yield_high = (tnx_dist >= -YH) & usable
move_comp = (move_pct <= M_LO) & usable
cell = yield_high & move_comp
print(f"\n  PITCHED CELL: TNX within {100*YH:.1f}% of 252d high AND "
      f"MOVE level pctile <= {M_LO:.0f}  ->  {int(cell.sum())} days, "
      f"{len(declusters(px.index[cell], 10, px.index))} episodes (10td gap)")
print("  cell years:", dict(pd.Series(1, index=px.index[cell]).groupby(
    px.index[cell].year).sum()))

# ---------------------------------------------------------------- both signs
H = 5
for veh, sgn in [("TLT", -1.0), ("TLT", +1.0), ("IEF", -1.0), ("IEF", +1.0)]:
    lbl = ("SHORT " if sgn < 0 else "LONG ") + veh
    r = vehicle_ret(px, [(veh, sgn)], H, 1)
    v = px.index[cell.values & r.notna().values]
    epi = declusters(v, 10, px.index)
    loc = local_control(px.index[r.notna().values], v)
    rows = [summarize(r.loc[epi].values, f"{lbl} episodes"),
            summarize(r.loc[loc].values, "CTRL local +/-126td"),
            summarize(r[r.notna()].values, "CTRL all days")]
    show(rows, f"BOTH-SIGN SCAN h={H}: {lbl}")

print("\n  NOTE: four sign/vehicle combinations were inspected. Any winner is "
      "charged max-of-4 (Sidak on 4 correlated tests is roughly a x2-3 p "
      "inflation; the pair TLT/IEF is ~0.95 correlated so effective k~2).")

# ---------------------------------------------------------------- gate attribution
print("\n=== STEP 3: GATE ATTRIBUTION (the conjunction must beat BOTH parents) ===")
for veh in ("TLT", "IEF"):
    for h in (3, 5, 10, 21):
        r = fwd_lag(px[veh], h, 1)
        ok = r.notna()
        def cell_mean(m, gap):
            d = px.index[m.values & ok.values]
            if len(d) == 0:
                return np.nan, 0
            e = declusters(d, gap, px.index)
            return 100 * float(np.nanmean(r.loc[e].values)), len(e)
        g = max(h, 10)
        joint, nj = cell_mean(cell, g)
        p1, n1 = cell_mean(yield_high, g)
        p2, n2 = cell_mean(move_comp, g)
        alld = 100 * float(r[ok].mean())
        print(f"  {veh} h={h:2d}: joint {joint:+.3f}% (N={nj:3d}) | "
              f"yield-high-only {p1:+.3f}% (N={n1:3d}) | "
              f"MOVE-comp-only {p2:+.3f}% (N={n2:3d}) | all days {alld:+.3f}%"
              f"   -> beats both parents? "
              f"{'YES' if (joint > p1 and joint > p2) else 'NO'}")

# ---------------------------------------------------------------- depth split at LIVE value
print("\n=== STEP 4: MOVE conditioner split at TODAY'S LIVE 44.4 pctile ===")
LIVE_MOVE = float(move_pct.iloc[-1])
bands = [(0, 20), (20, 40), (40, 50), (50, 60), (60, 80), (80, 101)]
for veh in ("TLT", "IEF"):
    for h in (5, 10):
        r = fwd_lag(px[veh], h, 1)
        ok = r.notna()
        base_d = px.index[yield_high.values & ok.values]
        base_e = declusters(base_d, max(h, 10), px.index)
        base = 100 * float(np.nanmean(r.loc[base_e].values))
        out = []
        for lo, hi in bands:
            m = yield_high & (move_pct >= lo) & (move_pct < hi)
            d = px.index[m.values & ok.values]
            if len(d) < 2:
                out.append({"band": f"[{lo},{hi})", "n_days": len(d), "n_epi": 0})
                continue
            e = declusters(d, max(h, 10), px.index)
            s = summarize(r.loc[e].values, f"[{lo},{hi})")
            s["gate_pp"] = round(s["mean_pct"] - base, 3)
            s["n_days"] = len(d)
            s["LIVE"] = "<<< 44.4" if lo <= LIVE_MOVE < hi else ""
            out.append(s)
        show(out, f"{veh} h={h} long: yield-high parent = {base:+.3f}%; "
                  f"by MOVE level pctile band")

# ---------------------------------------------------------------- era / midterm
print("\n=== STEP 5: era + midterm splits (episodes, h=5 and h=10, LONG TLT) ===")
for veh in ("TLT", "IEF"):
    for h in (5, 10):
        r = fwd_lag(px[veh], h, 1)
        d = px.index[cell.values & r.notna().values]
        e = declusters(d, max(h, 10), px.index)
        vals = r.loc[e].values
        show(era_split(e, vals), f"{veh} h={h} LONG era split")
        mid = (e.year % 4 == 2)
        show([summarize(vals[mid], f"midterm (N={int(mid.sum())})"),
              summarize(vals[~mid], f"non-midterm (N={int((~mid).sum())})")],
             f"{veh} h={h} LONG midterm split")

# ---------------------------------------------------------------- full battery, best sign
print("\n=== STEP 6: full battery on the pitched cell, both signs, h=5 and h=10 ===")
variants = {
    "TNX within 0.5% + MOVE<=50": (tnx_dist >= -0.005) & (move_pct <= 50) & usable,
    "TNX within 1.0% + MOVE<=50": cell,
    "TNX within 2.0% + MOVE<=50": (tnx_dist >= -0.02) & (move_pct <= 50) & usable,
    "TNX within 1.0% + MOVE<=40": (tnx_dist >= -YH) & (move_pct <= 40) & usable,
    "TNX within 1.0% + MOVE<=60": (tnx_dist >= -YH) & (move_pct <= 60) & usable,
    "TNX 1.0% + MOVE<=50 + 21dchg<-5%": cell & (move_21 < -0.05),
}
for h in (5, 10):
    battery(px, cell, [("TLT", 1.0)], h, f"C2 LONG TLT (orderly yield high)",
            3.0, variants=variants, min_gap=max(h, 10),
            event_kinds=("cpi", "ppi", "fomc_decision"))
    battery(px, cell, [("TLT", -1.0)], h, f"C2 SHORT TLT (orderly yield high)",
            3.0, variants=None, min_gap=max(h, 10),
            event_kinds=("cpi", "ppi", "fomc_decision"))

# ---------------------------------------------------------------- duration attribution
print("\n=== STEP 7: duration attribution (registry 2026-08-12) ===")
for h in (5, 10):
    rt, ri = fwd_lag(px["TLT"], h, 1), fwd_lag(px["IEF"], h, 1)
    ok = rt.notna() & ri.notna()
    d = px.index[cell.values & ok.values]
    e = declusters(d, max(h, 10), px.index)
    et, ei = rt.loc[e].values, ri.loc[e].values
    bt, bi = rt[ok].values, ri[ok].values
    ext, exi = et.mean() - bt.mean(), ei.mean() - bi.mean()
    sdt, sdi = rt[ok].std(), ri[ok].std()
    print(f"  h={h}: TLT excess {100*ext:+.3f}pp / sd {100*sdt:.3f}% = "
          f"{ext/sdt:+.4f}   IEF excess {100*exi:+.3f}pp / sd {100*sdi:.3f}% = "
          f"{exi/sdi:+.4f}   excess ratio {ext/exi if exi else np.nan:+.2f} vs "
          f"sd ratio {sdt/sdi:.2f}")
