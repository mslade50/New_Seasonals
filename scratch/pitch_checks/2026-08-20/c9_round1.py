"""C9 round 1: long TIP against short IEF ("breakevens") with the dollar on
the floor and gold bid.

Order of operations is the attack list's order, deliberately:
  0. measure the duration beta BEFORE constructing anything (an equal-dollar
     TIP/IEF pair is a duration SHORT wearing an inflation label)
  1. count the joint state, declustered, before designing a trade
  2. gate attribution: dollar-alone, gold-alone, joint
  3. leg attribution: which leg carries the spread, at every horizon
  4. does the residual behave like a breakeven proxy at all
  5. TIP's slow structural component
  6. cost
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["TIP", "IEF", "TLT", "GLD", "DX-Y.NYB", "USO", "CL=F", "SPY", "^TNX"]
px = close_panel(TK)
d = px.index
for t in TK:
    s = px[t].dropna()
    print(f"  {t:<10} {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

# ------------------------------------------------------------ 0. the beta
print("\n\n######## 0. DURATION BETA (measure, do not assume 1:1) ########")
rt = px["TIP"].pct_change()
ri = px["IEF"].pct_change()
both = pd.concat([rt, ri], axis=1).dropna()
both.columns = ["TIP", "IEF"]
BETA = float(np.polyfit(both["IEF"], both["TIP"], 1)[0])
print(f"  full-sample OLS beta(TIP on IEF) = {BETA:.4f}   "
      f"corr {both.corr().iloc[0,1]:.3f}   N={len(both)}")
print(f"  daily sd: TIP {100*both['TIP'].std():.4f}%  IEF {100*both['IEF'].std():.4f}%"
      f"  ratio {both['TIP'].std()/both['IEF'].std():.4f}")
for lo, hi in [("2003", "2011"), ("2012", "2019"), ("2020", "2026")]:
    sub = both.loc[lo:hi]
    b = float(np.polyfit(sub["IEF"], sub["TIP"], 1)[0])
    print(f"  beta {lo}-{hi}: {b:.4f}  (N={len(sub)})")
b5 = float(np.polyfit(both["IEF"].iloc[-1260:], both["TIP"].iloc[-1260:], 1)[0])
print(f"  beta trailing 5y: {b5:.4f}")

EQ = [("TIP", 1.0), ("IEF", -1.0)]                 # naive equal dollar
DN = [("TIP", 1.0), ("IEF", -round(BETA, 3))]      # duration-neutral residual
print(f"\n  EQUAL-DOLLAR pair  {EQ}")
print(f"  DURATION-NEUTRAL   {DN}   (beta {BETA:.3f})")

# ------------------------------------------------- 1. count the joint state
print("\n\n######## 1. COUNT THE JOINT STATE FIRST ########")
dx_r = pct_rank(px["DX-Y.NYB"], 21)
gl_r = pct_rank(px["GLD"], 21)
print(f"  TODAY  DXY 21d rank {dx_r.iloc[-1]:.2f}   GLD 21d rank {gl_r.iloc[-1]:.2f}")
m_dx = (dx_r <= 5).reindex(d).fillna(False)
m_gl = (gl_r >= 85).reindex(d).fillna(False)
m_jt = m_dx & m_gl
valid = vehicle_ret(px, DN, 10).notna()
for lbl, m in [("dollar alone (rank<=5)", m_dx), ("gold alone (rank>=85)", m_gl),
               ("JOINT", m_jt)]:
    s = d[m.values & valid.values]
    if len(s) == 0:
        print(f"  {lbl:<26} 0 days")
        continue
    e = declusters(s, 21, d)
    print(f"  {lbl:<26} {len(s):>4} days  {len(e):>3} episodes(gap21)  "
          f"years {sorted(set(e.year))}")

# ------------------------------------------------- 2/3. gate + leg attribution
print("\n\n######## 2+3. GATE ATTRIBUTION x LEG ATTRIBUTION ########")
legsets = {"TIP alone": [("TIP", 1.0)],
           "IEF alone (long)": [("IEF", 1.0)],
           "SHORT IEF leg": [("IEF", -1.0)],
           "pair equal-$": EQ,
           "pair duration-neutral": DN}
rows = []
for gname, m in [("dollar alone", m_dx), ("gold alone", m_gl), ("JOINT", m_jt)]:
    s = d[m.values]
    for h in (3, 5, 10):
        r0 = vehicle_ret(px, DN, h)
        e = declusters(pd.DatetimeIndex([x for x in s if not np.isnan(r0.get(x, np.nan))]),
                       21, d)
        if len(e) < 3:
            continue
        for lname, legs in legsets.items():
            r = vehicle_ret(px, legs, h)
            v = r.reindex(e).dropna()
            c = r.dropna()
            rows.append({"gate": gname, "legs": lname, "h": h, "N": len(v),
                         "mean_bps": round(10000 * v.mean(), 1),
                         "drift_bps": round(10000 * c.mean(), 1),
                         "excess_bps": round(10000 * (v.mean() - c.mean()), 1),
                         "hit": round(100 * (v > 0).mean(), 1),
                         "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
show(rows, "gate x leg x horizon (episode level, gap 21). BPS not percent.")

# ---------------------------------------- 4. is the residual a breakeven proxy?
print("\n\n######## 4. DOES THE RESIDUAL BEHAVE LIKE A BREAKEVEN? ########")
res = (px["TIP"].pct_change() - BETA * px["IEF"].pct_change()).dropna()
drivers = {"GLD": px["GLD"].pct_change(), "CL=F": px["CL=F"].pct_change(),
           "USO": px["USO"].pct_change(), "DXY": px["DX-Y.NYB"].pct_change(),
           "SPY": px["SPY"].pct_change(), "TNXlvl": px["^TNX"].diff()}
jt_days = d[m_jt.values]
print("  contemporaneous daily correlation of the TIP-beta*IEF residual with:")
for k, v in drivers.items():
    a = pd.concat([res, v], axis=1).dropna()
    aj = a.loc[a.index.intersection(jt_days)]
    print(f"    {k:<7} all days r={a.corr().iloc[0,1]:+.3f} (N={len(a)})   "
          f"joint-state days r={aj.corr().iloc[0,1]:+.3f} (N={len(aj)})")
print("\n  story requires: residual UP when gold UP and dollar DOWN.")

# 5-session forward residual vs the forward move of each driver, on trigger eps
print("\n  forward (h=5, lag=1) residual vs forward driver move, JOINT episodes:")
r5 = vehicle_ret(px, DN, 5)
e5 = declusters(pd.DatetimeIndex([x for x in jt_days
                                  if not np.isnan(r5.get(x, np.nan))]), 21, d)
for k, tkr in [("GLD", "GLD"), ("DXY", "DX-Y.NYB"), ("CL=F", "CL=F")]:
    fd = vehicle_ret(px, [(tkr, 1.0)], 5)
    a = pd.DataFrame({"res": r5.reindex(e5), "drv": fd.reindex(e5)}).dropna()
    if len(a) > 3:
        print(f"    {k:<7} r={a.corr().iloc[0,1]:+.3f} (N={len(a)})")

# ----------------------------------------- 5. TIP's slow structural component
print("\n\n######## 5. TIP's SLOW COMPONENT (carry / index lag) ########")
for h in (3, 5, 10):
    for lbl, legs in [("equal-$", EQ), ("duration-neutral", DN)]:
        r = vehicle_ret(px, legs, h).dropna()
        yr = r.groupby(r.index.year).mean() * 10000
        print(f"  h={h:<3} {lbl:<17} all-days drift {10000*r.mean():+.2f} bps  "
              f"| by-year bps: " +
              " ".join(f"{y}:{v:+.1f}" for y, v in yr.round(1).items()))
    break
r10 = vehicle_ret(px, DN, 10).dropna()
yr = (r10.groupby(r10.index.year).mean() * 10000).round(1)
print(f"\n  h=10 duration-neutral drift by year (bps): {dict(yr)}")

# ------------------------------------------------------------------ 6. cost
print("\n\n######## 6. COST ########")
print("  TIP round trip ~3 bps (spread ~1.5 bps + impact), IEF ~3 bps -> ~6 bps"
      " for the pair; a duration-neutral pair still crosses two spreads.")
for h in (3, 5, 10):
    r = vehicle_ret(px, DN, h)
    e = declusters(pd.DatetimeIndex([x for x in jt_days
                                     if not np.isnan(r.get(x, np.nan))]), 21, d)
    v = r.reindex(e).dropna()
    if len(v) >= 3:
        print(f"  h={h:<3} JOINT duration-neutral {10000*v.mean():+.2f} bps  "
              f"-> {10000*v.mean()/6:.2f}x cost (need >=5x)")
