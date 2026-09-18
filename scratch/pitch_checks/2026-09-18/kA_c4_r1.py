"""c4 round 1: SPY-hedged SHORT SVXY from the SEPTEMBER opex close to +h.

Entry MOC on the opex close (anchor = opex date, lag=0 -- the date is known in
advance, so this is the tradeable order), exit MOC +h. Vehicles:
  real SVXY post-break (2018-09..2025-09, 8 Septembers)
  spliced -0.5x SVS pre-break (2011-2017, 7 Septembers)  [era context]
  ^VIX spot 2000+ (master_prices starts 2000, not 1990) + its SPY residual
  SPY / IWM themselves (is it T3 in vol clothing?)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from kA_common import *  # noqa
import numpy as np
import pandas as pd

px = build_panel()
cal = px.index
vix = px["^VIX"]
vchg = vix / vix.shift(1) - 1
opex = pd.DatetimeIndex(sorted(set(load_events(["opex"])["date"]) & set(cal)))
opex = opex[opex < pd.Timestamp("2026-09-18")]
sep = opex[opex.month == 9]
post = pd.Series(cal >= POST, index=cal)
svxy_ok = pd.Series(cal >= pd.Timestamp("2011-10-10"), index=cal)
print(f"opex anchors {len(opex)} {opex[0].date()}..{opex[-1].date()}; September {len(sep)}")

COST = 12.0


def at(dates, off=0):
    p = cal.get_indexer(pd.DatetimeIndex(dates)) + off
    p = p[(p >= 0) & (p < len(cal))]
    return cal[p]


def hedged(veh, h, era_mask, lag=0):
    b = hedge_beta(px, veh, h, lag, era_mask)
    return b, vehicle_ret(px, [(veh, -1.0), ("SPY", b)], h, lag)


def vix_resid(h, lag=0):
    rv = vehicle_ret(px, [("^VIX", 1.0)], h, lag)
    rs = vehicle_ret(px, [("SPY", 1.0)], h, lag)
    ok = rv.notna() & rs.notna()
    b = np.polyfit(rs[ok].values, rv[ok].values, 1)[0]
    return b, rv - b * rs, rv


# ----------------------------------------------------------------- 1. the cell + controls
for h in (1, 2, 3, 4, 5):
    rows = []
    b, hs = hedged("SVXY", h, post)
    raw_short = -vehicle_ret(px, [("SVXY", 1.0)], h, 0)
    spy = vehicle_ret(px, [("SPY", 1.0)], h, 0)
    iwm = vehicle_ret(px, [("IWM", 1.0)], h, 0)
    sp = sep[sep >= POST]
    op = opex[(opex >= POST) & (opex.month != 9)]
    rows += [rec_row(hs.loc[sp].values, f"SEP hedged short SVXY post-break (b {b:.2f})", COST),
             rec_row(raw_short.loc[sp].values, "SEP raw short SVXY post-break"),
             rec_row(spy.loc[sp].values, "SEP SPY post-break (T3 side, long)"),
             rec_row(hs.loc[op].values, "CTRL other-month opex hedged, post-break", COST),
             rec_row(hs[post & hs.notna()].values, "CTRL all days hedged, post-break")]
    bp, hsp = hedged("SVS", h, pd.Series((cal < BREAK) & svxy_ok.values, index=cal))
    spre = sep[(sep < BREAK) & (sep >= pd.Timestamp("2011-10-10"))]
    opre = opex[(opex < BREAK) & (opex >= pd.Timestamp("2011-10-10")) & (opex.month != 9)]
    rows += [rec_row(hsp.loc[spre].values, f"SEP hedged short SVS PRE-break synth (b {bp:.2f})", COST),
             rec_row(hsp.loc[opre].values, "CTRL other-month opex, PRE-break synth", COST)]
    bv, res, rv = vix_resid(h)
    s00 = sep
    o00 = opex[opex.month != 9]
    rows += [rec_row(rv.loc[s00].values, "SEP ^VIX spot change 2000+ (long vol sign)"),
             rec_row(res.loc[s00].values, f"SEP ^VIX SPY-residual 2000+ (b {bv:.2f})"),
             rec_row(res.loc[o00].values, "CTRL other-month opex ^VIX residual"),
             rec_row(res[res.notna()].values, "CTRL all days ^VIX residual"),
             rec_row(spy.loc[s00].values, "SEP SPY 2000+"),
             rec_row(iwm.loc[s00].values, "SEP IWM 2000+")]
    show(rows, f"1. September opex close -> +{h}")

# ----------------------------------------------------------------- 2. year-by-year h=3
print("\n=== 2. September year by year, h=3 (entry opex close) ===")
b3, hs3 = hedged("SVXY", 3, post)
bp3, hsp3 = hedged("SVS", 3, pd.Series((cal < BREAK) & svxy_ok.values, index=cal))
bv3, res3, rv3 = vix_resid(3)
spy3 = vehicle_ret(px, [("SPY", 1.0)], 3, 0)
for x in sep:
    pre_path = vix.loc[x] / vix.shift(1).loc[at([x], -1)[0]] if False else np.nan
    p = cal.get_loc(x)
    crush3 = (vchg.iloc[p - 3:p + 0]).min()   # one-day VIX changes on opex-3..opex-1
    vts = px["^VIX"].iloc[p - 1] / px["^VIX3M"].iloc[p - 1] if not np.isnan(px["^VIX3M"].iloc[p - 1]) else np.nan
    hsv = hs3.get(x, np.nan) if x >= POST else hsp3.get(x, np.nan)
    print(f"  {x.date()} midterm={x.year%4==2}  SPY {100*spy3[x]:+6.2f}%  VIX {100*rv3[x]:+7.2f}%  "
          f"VIXres {100*res3[x]:+7.2f}%  hedged short {'SVXY' if x>=POST else 'SVS '} {100*hsv:+6.2f}%  "
          f"min 1d VIX chg opex-3..-1 {100*crush3:+6.1f}%  VIX/VIX3M(opex-1) {vts:.3f}")

# ----------------------------------------------------------------- 3. offset ladder around Sep opex
print("\n=== 3. placebo offset ladder: anchor = Sep opex + k, h=3 ===")
bs3, hss3 = hedged("SVS", 3, svxy_ok)  # full 2011+ spliced, one beta (era mix, context only)
for lbl, ser, dates in [("^VIX residual 2000+", res3, sep),
                        ("^VIX spot 2000+", rv3, sep),
                        ("hedged short SVXY post-break", hs3, sep[sep >= POST]),
                        ("hedged short SVS 2011+ (spliced)", hss3, sep[sep >= pd.Timestamp("2011-10-10")])]:
    rows = []
    for k in range(-5, 6):
        a = at(dates, k)
        v = ser.reindex(a).dropna().values
        w = int((v > 0).sum())
        rows.append({"k": k, "n": len(v), "mean_pct": 100 * v.mean(), "median_pct": 100 * np.median(v),
                     "hit": 100 * (v > 0).mean(), "rec": f"{w}-{len(v)-w}"})
    df = pd.DataFrame(rows)
    df["rank"] = df["mean_pct"].rank(ascending=False).astype(int)
    print(f"\n-- {lbl} --")
    print(df.round(3).to_string(index=False))
    print(f"   TRUE k=0 ranks {int(df.loc[df.k == 0, 'rank'].iloc[0])} of {len(df)}")

# ----------------------------------------------------------------- 4. month-of-year ladder
print("\n=== 4. month ladder at the opex close, h=3 ===")
rows = []
for mo in range(1, 13):
    om = opex[opex.month == mo]
    r1 = res3.reindex(om).dropna().values
    r2 = hs3.reindex(om[om >= POST]).dropna().values
    r3 = spy3.reindex(om).dropna().values
    rows.append({"month": mo, "n_vix": len(r1), "vixres_mean": 100 * r1.mean(), "vixres_hit": 100 * (r1 > 0).mean(),
                 "n_svxy": len(r2), "hedged_short_svxy": 100 * r2.mean() if len(r2) else np.nan,
                 "hs_hit": 100 * (r2 > 0).mean() if len(r2) else np.nan, "spy_mean": 100 * r3.mean()})
df = pd.DataFrame(rows)
df["rank_vixres"] = df["vixres_mean"].rank(ascending=False).astype(int)
df["rank_hs"] = df["hedged_short_svxy"].rank(ascending=False).astype(int)
print(df.round(3).to_string(index=False))

# ----------------------------------------------------------------- 5. conditioning on the pre-opex VIX path (all months)
print("\n=== 5. all-opex conditioning: one-day crush >=10% on opex-3..opex-1, h=3 ===")
for lbl, ser, base in [("hedged short SVXY post-break", hs3, opex[opex >= POST]),
                       ("^VIX residual 2000+", res3, opex)]:
    cr, nc = [], []
    for x in base:
        p = cal.get_loc(x)
        (cr if vchg.iloc[p - 3:p].min() <= -0.10 else nc).append(x)
    cr, nc = pd.DatetimeIndex(cr), pd.DatetimeIndex(nc)
    crs = cr[cr.month == 9]
    show([rec_row(ser.reindex(cr).values, f"{lbl}: opex after crush"),
          rec_row(ser.reindex(nc).values, f"{lbl}: opex no crush"),
          rec_row(ser.reindex(crs).values, f"{lbl}: SEPT opex after crush (JOINT)")])
    print("   joint dates:", [str(d.date()) for d in crs])

print("\n=== 5b. VIX/VIX3M at opex-1 split, ^VIX residual h=3, Sept vs rest ===")
vts = px["^VIX"] / px["^VIX3M"]
for lbl, dates in [("Sept", sep), ("other months", opex[opex.month != 9])]:
    d = pd.DatetimeIndex([x for x in dates if not np.isnan(vts.iloc[cal.get_loc(x) - 1])])
    lo = pd.DatetimeIndex([x for x in d if vts.iloc[cal.get_loc(x) - 1] < 0.85])
    hi = d.difference(lo)
    show([rec_row(res3.reindex(lo).values, f"{lbl} VIX/VIX3M<0.85"),
          rec_row(res3.reindex(hi).values, f"{lbl} VIX/VIX3M>=0.85")])

# ----------------------------------------------------------------- 6. midterm + era, ^VIX residual h=3
print("\n=== 6. Sept ^VIX residual h=3: era + midterm ===")
v = res3.reindex(sep)
show([rec_row(v[v.index < "2018-01-01"].values, "pre-2018"),
      rec_row(v[v.index >= "2018-01-01"].values, "2018+"),
      rec_row(v[v.index.year % 4 == 2].values, "midterm"),
      rec_row(v[v.index.year % 4 != 2].values, "non-midterm")])
print("  concentration:", signed_concentration(v.dropna().index, v.dropna().values))
w = hs3.reindex(sep[sep >= POST]).dropna()
print("  hedged short SVXY post-break concentration:", signed_concentration(w.index, w.values))
