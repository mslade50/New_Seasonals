"""C1 round 1 -- long SVXY (short vol) across the Jackson Hole speech.

Mechanism claim: the speech carries an event premium that is sold once
delivered, so implied vol falls from the JH-1 close through JH+3.

Order of attack:
  A. does implied vol actually fall?  VIX and VIX3M level paths around the
     anchor, 2000+/2006+, against the same-span unconditional baseline.
  B. is the move where the mechanism says it is?  speech day is 10:00 ET, so
     an event-premium story predicts an INTRADAY drop on JH day, not a gap.
  C. the tradeable vehicle: long SVXY, entry MOC at the JH-1 close.
  D. offset placebo ladder (the 3-for-3 killer in this repo).
  E. SVXY leverage-regime split (-1x pre 2018-02-27, -0.5x after) and the
     midterm split, which has inverted JH in 4 of 6 vehicles.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

JH = load_events(["jackson_hole"])["date"]
JH = JH[JH.dt.year <= 2025]  # 2026 has not happened
print(f"JH anchors used: {len(JH)}  {JH.dt.year.min()}..{JH.dt.year.max()}")

pxd = load_prices(["^VIX", "^VIX3M", "SPY", "SVXY", "UVXY"])


def anchor_positions(idx, dates):
    """Index position of each anchor DAY, exact match required."""
    pos = pd.Series(range(len(idx)), index=idx)
    out, miss = [], []
    for d in dates:
        p = pos.get(d)
        if p is None:
            miss.append(str(d.date()))
        else:
            out.append(int(p))
    if miss:
        print(f"  anchors not in index (skipped): {miss}")
    return out


# ---------------------------------------------------------------- A. the VIX
print("\n" + "=" * 78)
print("A. does implied vol fall around the speech?  (level change from the")
print("   JH-1 close, in VIX points and in %, vs the same-span baseline)")
print("=" * 78)

for name, s in (("^VIX", pxd["^VIX"]["Close"]), ("^VIX3M", pxd["^VIX3M"]["Close"])):
    s = s.dropna()
    idx = s.index
    ap = anchor_positions(idx, JH)
    rows = []
    for k in range(0, 6):
        d_pts, d_pct = [], []
        for p in ap:
            i0, i1 = p - 1, p + k
            if i0 < 0 or i1 >= len(s):
                continue
            d_pts.append(s.iloc[i1] - s.iloc[i0])
            d_pct.append(s.iloc[i1] / s.iloc[i0] - 1.0)
        span = k + 1
        base = (s.shift(-span) / s - 1.0).dropna()
        base_pts = (s.shift(-span) - s).dropna()
        rows.append({
            "leg": f"JH-1 -> JH+{k}", "n": len(d_pts),
            "d_vix_pts": round(float(np.mean(d_pts)), 3),
            "d_vix_pct": round(100 * float(np.mean(d_pct)), 2),
            "hit_down_%": round(100 * float(np.mean(np.array(d_pct) < 0)), 1),
            "sign_p_down": round(sign_test(int((np.array(d_pct) < 0).sum()), len(d_pct)), 4),
            "base_pts": round(float(base_pts.mean()), 3),
            "base_pct": round(100 * float(base.mean()), 2),
            "excess_pct": round(100 * float(np.mean(d_pct)) - 100 * float(base.mean()), 2),
        })
    show(rows, f"{name} path around Jackson Hole (N anchors in index above)")

# ------------------------------------------------- B. gap vs intraday on JH day
print("\n" + "=" * 78)
print("B. mechanism location. Speech is 10:00 ET => an event-premium story")
print("   predicts the VIX drop lands INTRADAY on JH day, not in the gap.")
print("=" * 78)
v = pxd["^VIX"].dropna(subset=["Open", "Close"])
idx = v.index
ap = anchor_positions(idx, JH)
gap, intr, tot = [], [], []
for p in ap:
    if p - 1 < 0:
        continue
    pc = v["Close"].iloc[p - 1]
    o = v["Open"].iloc[p]
    c = v["Close"].iloc[p]
    gap.append(o / pc - 1.0)
    intr.append(c / o - 1.0)
    tot.append(c / pc - 1.0)
gap, intr, tot = map(np.array, (gap, intr, tot))
print(f"  N={len(tot)}  JH-day VIX total {100*tot.mean():+.2f}%  "
      f"= gap {100*gap.mean():+.2f}%  + intraday {100*intr.mean():+.2f}%")
if tot.mean() != 0:
    print(f"  gap share of the total move: {100*gap.mean()/tot.mean():.0f}%   "
          f"intraday share: {100*intr.mean()/tot.mean():.0f}%")
print(f"  intraday down-days {int((intr<0).sum())}/{len(intr)}  "
      f"sign p = {sign_test(int((intr<0).sum()), len(intr)):.4f}")

# --------------------------------------------------------- C. the vehicle
print("\n" + "=" * 78)
print("C. the tradeable object: LONG SVXY, signal JH-2 close, entry MOC")
print("   JH-1 close, exit h sessions later (h=4 -> JH+3).")
print("=" * 78)
px = pd.DataFrame({t: pxd[t]["Close"] for t in ("SVXY", "SPY", "UVXY")}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

sig_pos = []
for d in JH:
    p = pos.get(d)
    if p is None or p - 2 < 0:
        continue
    sig_pos.append(p - 2)
sig_dates = idx[sig_pos]
print(f"  SVXY-era JH anchors: N={len(sig_dates)}  "
      f"{sig_dates.year.min()}..{sig_dates.year.max()}")

mask = pd.Series(False, index=idx)
mask.loc[sig_dates] = True

battery(px, mask, [("SVXY", 1.0)], h=4,
        title="C1: long SVXY, JH-1 close entry, hold 4td (JH+3)",
        cost_bps=5.0, min_gap=60, event_kinds=("nfp",))

for h in (1, 2, 3, 4, 5, 6, 8, 10):
    r = summarize(vehicle_ret(px, [("SVXY", 1.0)], h, 1).loc[sig_dates].values,
                  f"SVXY h={h}")
    allv = vehicle_ret(px, [("SVXY", 1.0)], h, 1).dropna()
    r["drift_pct"] = round(100 * allv.mean(), 3)
    r["excess_pct"] = round(r["mean_pct"] - 100 * allv.mean(), 3)
    r["sign_p"] = round(sign_test(int((vehicle_ret(px, [("SVXY", 1.0)], h, 1)
                                       .loc[sig_dates].values > 0).sum()),
                                  len(sig_dates)), 4)
    print("  " + " ".join(f"{k}={v}" for k, v in r.items()))

# SPY-beta-hedged residual (the 2026-08-21 method): SVXY is ~ -0.5x SPY beta
print("\n  SPY-beta-hedged residual (SVXY minus b*SPY, b from trailing 252d):")
r_svxy = px["SVXY"].pct_change(fill_method=None)
r_spy = px["SPY"].pct_change(fill_method=None)
beta = (r_svxy.rolling(252).cov(r_spy) / r_spy.rolling(252).var()).shift(1)
for h in (2, 4, 6):
    raw = vehicle_ret(px, [("SVXY", 1.0)], h, 1)
    spyr = vehicle_ret(px, [("SPY", 1.0)], h, 1)
    resid = raw - beta * spyr
    c = summarize(resid.loc[sig_dates].dropna().values, f"resid h={h} COND")
    b = summarize(resid.dropna().values, f"resid h={h} ALL")
    print(f"   h={h}: cond {c['mean_pct']:+.3f}% (N={c['n']}, t={c['t']:.2f})  "
          f"all {b['mean_pct']:+.3f}%  excess {c['mean_pct']-b['mean_pct']:+.3f}pp  "
          f"mean beta at anchors {beta.loc[sig_dates].mean():.2f}")

# --------------------------------------------------- D. offset placebo ladder
print("\n" + "=" * 78)
print("D. OFFSET PLACEBO LADDER. Signal date = JH-2+k for k in -12..+12.")
print("   A plateau kills; only a spike at k=0 survives.")
print("=" * 78)
for h in (4, 2):
    ret = vehicle_ret(px, [("SVXY", 1.0)], h, 1)
    rows = []
    for k in range(-12, 13):
        sp = [p - 2 + k for d in JH
              if (p := pos.get(d)) is not None and 0 <= p - 2 + k < len(idx)]
        dts = idx[sp]
        vals = ret.loc[dts].dropna().values
        if len(vals) == 0:
            continue
        rows.append({"k": k, "n": len(vals), "mean_pct": round(100 * vals.mean(), 3),
                     "hit": round(100 * (vals > 0).mean(), 1)})
    df = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
    rank = int(df.index[df["k"] == 0][0]) + 1
    print(f"\n  h={h}: TRUE ANCHOR k=0 RANKS {rank} of {len(df)}")
    print(df.to_string(index=False))

# -------------------------------------- E. leverage regime + midterm splits
print("\n" + "=" * 78)
print("E. SVXY leverage regime (-1x before 2018-02-27, -0.5x after) and the")
print("   midterm split.")
print("=" * 78)
h = 4
ret = vehicle_ret(px, [("SVXY", 1.0)], h, 1)
vals = ret.loc[sig_dates].values
yrs = sig_dates.year
cut = pd.Timestamp("2018-02-27")
m1x = sig_dates < cut
rows = [summarize(vals[m1x], f"-1x era (<=2017), N={int(m1x.sum())}"),
        summarize(vals[~m1x], f"-0.5x era (2018+), N={int((~m1x).sum())}")]
mt = (yrs % 4 == 2)
rows += [summarize(vals[mt], f"MIDTERM years, N={int(mt.sum())}"),
         summarize(vals[~mt], f"non-midterm, N={int((~mt).sum())}")]
show(rows, "C1 splits, h=4")
print("  per-year h=4 SVXY returns:")
for d, val in zip(sig_dates, vals):
    print(f"    {d.year}  signal {d.date()}  {100*val:+7.2f}%"
          f"{'   [MIDTERM]' if d.year % 4 == 2 else ''}")
