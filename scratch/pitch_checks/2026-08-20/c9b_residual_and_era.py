"""C9 follow-up: what IS the TIP/IEF residual, and does the cell survive
era + concentration once the leg attribution has taken its cut?

  A. SPY-neutralise the duration-neutral residual and see what alpha is left
  B. era split + concentration on the JOINT cell AND on its dollar-alone parent
  C. drop the 2008-09 TIPS liquidity dislocation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TIP", "IEF", "GLD", "DX-Y.NYB", "SPY", "CL=F"])
d = px.index
BETA = 0.698
DN = [("TIP", 1.0), ("IEF", -BETA)]
EQ = [("TIP", 1.0), ("IEF", -1.0)]

dx_r = pct_rank(px["DX-Y.NYB"], 21)
gl_r = pct_rank(px["GLD"], 21)
m_dx = (dx_r <= 5).reindex(d).fillna(False)
m_jt = (m_dx & (gl_r >= 85).reindex(d).fillna(False))


def eps(mask, h):
    r = vehicle_ret(px, DN, h)
    s = [x for x in d[mask.values] if not np.isnan(r.get(x, np.nan))]
    return declusters(pd.DatetimeIndex(s), 21, d)


print("######## A. SPY-NEUTRALISE THE 'BREAKEVEN' RESIDUAL ########")
for h in (3, 5, 10):
    rp = vehicle_ret(px, DN, h)
    rs = vehicle_ret(px, [("SPY", 1.0)], h)
    rg = vehicle_ret(px, [("GLD", 1.0)], h)
    a = pd.concat([rp, rs, rg], axis=1).dropna()
    a.columns = ["pair", "spy", "gld"]
    bspy = float(np.polyfit(a["spy"], a["pair"], 1)[0])
    bgld = float(np.polyfit(a["gld"], a["pair"], 1)[0])
    e = eps(m_jt, h)
    e = pd.DatetimeIndex([x for x in e if x in a.index])
    raw = a.loc[e, "pair"]
    alpha_spy = a.loc[e, "pair"] - bspy * a.loc[e, "spy"]
    alpha_g = a.loc[e, "pair"] - bgld * a.loc[e, "gld"]
    print(f"  h={h:<3} N={len(e)}  raw {10000*raw.mean():+.2f} bps | "
          f"SPY beta {bspy:+.3f} -> SPY-neutral {10000*alpha_spy.mean():+.2f} bps"
          f" (hit {100*(alpha_spy>0).mean():.0f}%) | "
          f"GLD beta {bgld:+.3f} -> GLD-neutral {10000*alpha_g.mean():+.2f} bps")

print("\n\n######## B. ERA + CONCENTRATION ########")
for lbl, mk in [("JOINT (dollar+gold)", m_jt), ("PARENT dollar-alone", m_dx)]:
    print(f"\n--- {lbl} ---")
    for h in (3, 5, 10):
        r = vehicle_ret(px, DN, h)
        e = eps(mk, h)
        v = r.reindex(e).dropna()
        c = r.dropna()
        print(f"\n h={h}  N={len(v)}  mean {10000*v.mean():+.2f} bps  "
              f"excess {10000*(v.mean()-c.mean()):+.2f} bps  "
              f"hit {100*(v>0).mean():.1f}%  sign p "
              f"{sign_test(int((v>0).sum()), len(v)):.3f}")
        rows = era_split(v.index, v.values)
        for r_ in rows:
            if r_.get("n"):
                print(f"    {r_['label']:<10} N={r_['n']:<3} "
                      f"{100*r_['mean_pct']:+.2f} bps  hit {r_['hit']:.1f}%")
        print("   ", cluster_note(v.index, v.values))

print("\n\n######## C. DROP THE 2008-09 TIPS LIQUIDITY DISLOCATION ########")
for h in (3, 5, 10):
    r = vehicle_ret(px, DN, h)
    e = eps(m_jt, h)
    v = r.reindex(e).dropna()
    keep = v[~v.index.year.isin([2008, 2009])]
    drop = v[v.index.year.isin([2008, 2009])]
    print(f"  h={h:<3} full N={len(v)} {10000*v.mean():+.2f} bps | "
          f"ex-2008/09 N={len(keep)} {10000*keep.mean():+.2f} bps | "
          f"2008/09 only N={len(drop)} {10000*drop.mean():+.2f} bps")
    print(f"       ex-2008/09 -> {10000*keep.mean()/6:.2f}x the 6 bps round trip")
