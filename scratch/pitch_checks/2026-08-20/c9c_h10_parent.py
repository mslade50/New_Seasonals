"""C9 closer: h=10 is the only horizon that survived the 2008-09 drop, so
test whether the GOLD gate earns anything there once the crisis is removed,
and whether the parent (dollar alone) already has it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TIP", "IEF", "GLD", "DX-Y.NYB", "SPY"])
d = px.index
DN = [("TIP", 1.0), ("IEF", -0.698)]
dx_r = pct_rank(px["DX-Y.NYB"], 21)
gl_r = pct_rank(px["GLD"], 21)
m_dx = (dx_r <= 5).reindex(d).fillna(False)
m_gl = (gl_r >= 85).reindex(d).fillna(False)
m_jt = m_dx & m_gl

for h in (3, 5, 10):
    r = vehicle_ret(px, DN, h)
    c = r.dropna()
    print(f"\n### h={h}  (all-days duration-neutral drift {10000*c.mean():+.2f} bps)")
    for lbl, m in [("dollar alone", m_dx), ("gold alone", m_gl), ("JOINT", m_jt)]:
        e = declusters(pd.DatetimeIndex(
            [x for x in d[m.values] if not np.isnan(r.get(x, np.nan))]), 21, d)
        v = r.reindex(e).dropna()
        k = v[~v.index.year.isin([2008, 2009])]
        print(f"  {lbl:<13} full N={len(v):>3} {10000*v.mean():+7.2f} bps "
              f"(excess {10000*(v.mean()-c.mean()):+6.2f}) | "
              f"ex-08/09 N={len(k):>3} {10000*k.mean():+7.2f} bps "
              f"hit {100*(k>0).mean():.0f}% sign p "
              f"{sign_test(int((k>0).sum()), len(k)):.3f} "
              f"-> {10000*k.mean()/6:.2f}x cost")
    # gate value, ex-crisis
    def m_ex(mask):
        e = declusters(pd.DatetimeIndex(
            [x for x in d[mask.values] if not np.isnan(r.get(x, np.nan))]), 21, d)
        v = r.reindex(e).dropna()
        return v[~v.index.year.isin([2008, 2009])]
    a, b = m_ex(m_jt), m_ex(m_dx)
    print(f"  GOLD GATE VALUE ex-08/09: joint {10000*a.mean():+.2f} - parent "
          f"{10000*b.mean():+.2f} = {10000*(a.mean()-b.mean()):+.2f} bps "
          f"for {len(b)-len(a)} fewer episodes")

print("\n\n### era stability of the ex-crisis h=10 cell (the only survivor) ###")
r = vehicle_ret(px, DN, 10)
e = declusters(pd.DatetimeIndex(
    [x for x in d[m_jt.values] if not np.isnan(r.get(x, np.nan))]), 21, d)
v = r.reindex(e).dropna()
k = v[~v.index.year.isin([2008, 2009])]
show(era_split(k.index, k.values), "JOINT h=10 ex-2008/09")
print(" ", cluster_note(k.index, k.values))
by = (k.groupby(k.index.year).mean() * 10000).round(1)
print("  by-year bps:", dict(by))
print("  episodes:", ", ".join(f"{x.date()}:{10000*val:+.0f}" for x, val in k.items()))
