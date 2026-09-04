"""C8 round 1 - long EEM against short EFA at a 63-day relative extreme.

Live premise: EEM-EFA 63d spread -7.57pp, PIT trailing-252 pctile 1.6
(full-sample 6.6); EEM 63d rank 6.3 vs EFA 59.1, SPY 23.8; EEM -7.16% off its
52w high, EFA -0.61%; FXI-EEM 63d +4.24pp (PIT 99.6).

Round-1 obligations discharged here:
  0. PREMISE re-derivation, PIT vs full-sample on the VALID span (the 08-18
     lookahead trap; note the naive full-sample form over a union calendar is
     deflated by the pre-inception NaN rows).
  1. battery() on the equal-dollar pair, h=5, rung ladder + gate variants.
  2. BETA-NEUTRAL residual on a POINT-IN-TIME trailing-252d beta.
  3. LEG ATTRIBUTION (the exact test that killed the 2026-08-19 EFA/SPY pair):
     each leg against its OWN drift, plus the naked long EEM.
  4. DOLLAR TEST (2026-08-20 registry): regress the pair's forward return on
     DXY's forward return over the same window, report the residual.
  5. ERA stability (pre/post 2018), midterm split, fragility-dial split.
  6. FXI sub-cell: is the EEM leg actually a China bet?
  7. tape over-selection, cost, concentration, JH-in-window.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = pd.Timestamp("2026-08-24")
NAMES = ["EEM", "EFA", "FXI", "SPY", "EWJ", "EWZ", "INDA", "DX-Y.NYB", "UUP"]

px_all = load_prices(NAMES)
spy = px_all["SPY"]["Close"].dropna()
CAL = spy.index[spy.index <= BAR]
px = pd.DataFrame({t: px_all[t]["Close"] for t in NAMES}).reindex(CAL)
for t in NAMES:
    s = px_all[t]["Close"].dropna()
    print(f"  {t}: {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")


def rn(t, n):
    return _valid_pct_change(px_all[t]["Close"].dropna(), n).reindex(CAL)


def dist_hi(t, look=252):
    c = px_all[t]["Close"].dropna()
    return rolling_on_valid(c, lambda x: x / x.rolling(look).max() - 1.0).reindex(CAL)


sp63 = rn("EEM", 63) - rn("EFA", 63)
pit63 = rolling_on_valid(sp63, lambda x: x.rolling(252).rank(pct=True) * 100)

print("\n" + "=" * 100)
print("0. PREMISE re-derivation")
print("=" * 100)
v = sp63.dropna()
print(f"  EEM r63 {100*rn('EEM',63).iloc[-1]:+.2f}%  EFA r63 {100*rn('EFA',63).iloc[-1]:+.2f}%"
      f"  spread {100*sp63.iloc[-1]:+.2f}pp")
print(f"  spread PIT-252 pctile {pit63.iloc[-1]:.2f}   FULL-SAMPLE pctile on the "
      f"VALID span {100*(v <= v.iloc[-1]).mean():.2f}  (n_valid={len(v)}; the naive "
      f"union-calendar form gives {100*(sp63 <= sp63.iloc[-1]).mean():.2f}, deflated by "
      f"{len(sp63)-len(v)} pre-inception NaN rows)")
for t in ["EEM", "EFA", "FXI", "SPY"]:
    print(f"  {t}: off 52wh {100*dist_hi(t).iloc[-1]:+.2f}%   r63 rank "
          f"{pct_rank(px_all[t]['Close'].dropna(),63).reindex(CAL).iloc[-1]:.1f}")
fxi_eem = rn("FXI", 63) - rn("EEM", 63)
print(f"  FXI-EEM 63d {100*fxi_eem.iloc[-1]:+.2f}pp  PIT "
      f"{rolling_on_valid(fxi_eem, lambda x: x.rolling(252).rank(pct=True)*100).iloc[-1]:.1f}")

# ------------------------------------------------------------- triggers
TRIG = {
    "A PIT63 <= 1": (pit63 <= 1.0).fillna(False),
    "B PIT63 <= 2 (pitched)": (pit63 <= 2.0).fillna(False),
    "C PIT63 <= 5": (pit63 <= 5.0).fillna(False),
    "D PIT63 <= 10": (pit63 <= 10.0).fillna(False),
    "E PIT63<=2 & EFA off-52wh >= -2% (LIVE)": (pit63 <= 2.0) & (dist_hi("EFA") >= -0.02),
    "F PIT63<=2 & EEM off-52wh <= -5% (LIVE)": (pit63 <= 2.0) & (dist_hi("EEM") <= -0.05),
    "G PIT63<=2 & FXI-EEM 63d > 0 (LIVE)": (pit63 <= 2.0) & (fxi_eem > 0),
}
print("\n  trigger day counts:")
for k, m in TRIG.items():
    m = m.reindex(CAL, fill_value=False).fillna(False)
    print(f"    {k:45s} n_days={int(m.sum()):4d}  live={bool(m.iloc[-1])}")
MAIN = TRIG["B PIT63 <= 2 (pitched)"].reindex(CAL, fill_value=False).fillna(False)

battery(px, MAIN, [("EEM", 1.0), ("EFA", -1.0)], h=5,
        title="C8 equal-dollar EEM long / EFA short, PIT63 spread <= 2",
        cost_bps=4.0, variants=TRIG,
        event_kinds=("jackson_hole", "fomc_decision", "cpi"))

# ------------------------------------------- 2. beta-neutral + naked, horizons
print("\n" + "=" * 100)
print("2. equal-dollar vs BETA-NEUTRAL (PIT 252d beta) vs NAKED long EEM")
print("=" * 100)
re_, rf = px["EEM"].pct_change(), px["EFA"].pct_change()
beta = (re_.rolling(252).cov(rf) / rf.rolling(252).var()).reindex(CAL)
print(f"  live beta EEM-on-EFA (252d) = {beta.iloc[-1]:.3f}  median {beta.median():.3f}"
      f"  range [{beta.min():.2f}, {beta.max():.2f}]")
rows = []
for h in (1, 2, 3, 5, 10):
    eq = vehicle_ret(px, [("EEM", 1.0), ("EFA", -1.0)], h)
    rs = fwd_lag(px["EEM"], h) - beta * fwd_lag(px["EFA"], h)
    nk = fwd_lag(px["EEM"], h)
    valid = eq.notna() & rs.notna()
    epi = declusters(CAL[MAIN.values & valid.values], h, CAL[valid.values])
    for lbl, ser in (("equal-dollar", eq), ("beta-neutral", rs), ("naked EEM", nk)):
        r = summarize(ser.loc[epi].values, f"h={h} {lbl}")
        b = ser[valid].mean()
        r["ctl_all_pct"] = round(100 * b, 3)
        r["edge_pct"] = round(r["mean_pct"] - 100 * b, 3)
        rows.append(r)
show(rows, "2. three vehicles, episode level")

# --------------------------------------------------------- 3. leg attribution
print("\n" + "=" * 100)
print("3. LEG ATTRIBUTION - the test that killed the 2026-08-19 EFA/SPY pair")
print("=" * 100)
for h in (1, 3, 5, 10):
    eq = vehicle_ret(px, [("EEM", 1.0), ("EFA", -1.0)], h)
    valid = eq.notna()
    trig = CAL[MAIN.values & valid.values]
    epi = declusters(trig, h, CAL[valid.values])
    span = (CAL >= trig[0]) & (CAL <= trig[-1]) & valid.values
    e_f, f_f = fwd_lag(px["EEM"], h), fwd_lag(px["EFA"], h)
    e_c, f_c = e_f[span].mean(), f_f[span].mean()
    e_t, f_t = e_f.loc[epi].mean(), f_f.loc[epi].mean()
    long_c, short_c = 100 * (e_t - e_c), 100 * (f_c - f_t)
    tot = long_c + short_c
    print(f"  h={h:2d} N={len(epi):3d}  LONG EEM {100*e_t:+.3f}% vs own drift "
          f"{100*e_c:+.3f}% -> {long_c:+.3f}pp  |  SHORT EFA {100*(-f_t):+.3f}% vs "
          f"{100*(-f_c):+.3f}% -> {short_c:+.3f}pp"
          + (f"   long share {100*long_c/tot:+.0f}% / short share {100*short_c/tot:+.0f}%"
             if abs(tot) > 1e-12 else ""))

# ------------------------------------------------------------ 4. dollar test
print("\n" + "=" * 100)
print("4. DOLLAR TEST (2026-08-20 registry: country pairs reduce to a dollar bet)")
print("=" * 100)
for h in (3, 5, 10):
    eq = vehicle_ret(px, [("EEM", 1.0), ("EFA", -1.0)], h)
    dx = fwd_lag(px["DX-Y.NYB"], h)
    valid = eq.notna() & dx.notna()
    epi = declusters(CAL[MAIN.values & valid.values], h, CAL[valid.values])
    sub = pd.DataFrame({"p": eq.loc[epi], "d": dx.loc[epi]}).dropna()
    if len(sub) < 5:
        continue
    b, a = np.polyfit(sub["d"], sub["p"], 1)
    print(f"  h={h:2d} N={len(sub):3d}  slope on DXY fwd {b:+.3f}  corr "
          f"{sub['p'].corr(sub['d']):+.3f}  raw mean {100*sub['p'].mean():+.3f}%  "
          f"DXY-RESIDUAL alpha {100*a:+.3f}%")
    allv = pd.DataFrame({"p": eq[valid], "d": dx[valid]}).dropna()
    b2, a2 = np.polyfit(allv["d"], allv["p"], 1)
    print(f"        all-days slope {b2:+.3f} corr {allv['p'].corr(allv['d']):+.3f} "
          f"-> the pair is {'dollar-driven' if abs(allv['p'].corr(allv['d']))>0.3 else 'not chiefly dollar-driven'}")

# ----------------------------------------------------- 5. era / regime splits
print("\n" + "=" * 100)
print("5. ERA, MIDTERM and FRAGILITY-DIAL splits (episodes, h=5)")
print("=" * 100)
h = 5
eq = vehicle_ret(px, [("EEM", 1.0), ("EFA", -1.0)], h)
valid = eq.notna()
epi = declusters(CAL[MAIN.values & valid.values], h, CAL[valid.values])
vals = eq.loc[epi].values
show(era_split(epi, vals), "pre/post 2018")
show(era_split(epi, vals, cut="2013-01-01"), "pre/post 2013 (second cut)")
mid = np.array([d.year % 4 == 2 for d in epi])
show([summarize(vals[mid], f"midterm (N={int(mid.sum())})"),
      summarize(vals[~mid], f"non-midterm (N={int((~mid).sum())})")], "cycle split")
fp = ROOT / "data" / "rd2_fragility.parquet"
if fp.exists():
    fg = pd.read_parquet(fp)
    fg.index = pd.to_datetime(fg.index)
    ma = fg["63d"].rolling(10).mean()
    d = ma.reindex(epi).values
    ok = ~np.isnan(d)
    print(f"  dial coverage on episodes: {int(ok.sum())}/{len(epi)}  "
          f"(pre-2016 has no dial; 2016..2026-07-02 is the RECOMPUTE vintage, "
          f"which is what this split uses)")
    if ok.sum() > 4:
        hi = ok & (d >= 65)
        show([summarize(vals[hi], f"dial ma10(63d) >= 65 (N={int(hi.sum())})"),
              summarize(vals[ok & ~hi], f"dial < 65 (N={int((ok & ~hi).sum())})")],
             "fragility split (today's dial = 89.5)")
        print(f"  episodes ever seen with dial >= 85 (today 89.5): {int((ok & (d>=85)).sum())}")

# ---------------------------------------------------------------- 6. FXI cell
print("\n" + "=" * 100)
print("6. Is the EEM leg a China bet? (FXI-EEM 63d > 0 today, PIT 99.6)")
print("=" * 100)
rows = []
for lbl, legs in (("long EEM / short EFA", [("EEM", 1.0), ("EFA", -1.0)]),
                  ("long EEM / short FXI", [("EEM", 1.0), ("FXI", -1.0)]),
                  ("long FXI / short EFA", [("FXI", 1.0), ("EFA", -1.0)]),
                  ("naked long EEM", [("EEM", 1.0)]),
                  ("naked long EFA", [("EFA", 1.0)])):
    ser = vehicle_ret(px, legs, 5)
    vv = ser.notna()
    e = declusters(CAL[MAIN.values & vv.values], 5, CAL[vv.values])
    r = summarize(ser.loc[e].values, lbl)
    r["edge_pct"] = round(r["mean_pct"] - 100 * ser[vv].mean(), 3)
    rows.append(r)
show(rows, "6. vehicle comparison on the same trigger, h=5")

# ------------------------------------------------------- 7. tape / cost notes
sma200 = rolling_on_valid(spy, lambda x: x.rolling(200).mean()).reindex(CAL)
above = px["SPY"] > sma200
trig = CAL[MAIN.values & valid.values]
print(f"\n7. TAPE over-selection: SPY above 200d on {100*above.loc[trig].mean():.1f}% of "
      f"trigger days vs base {100*above[valid].mean():.1f}%")
print("\nDONE C8 round 1")
