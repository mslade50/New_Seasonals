"""C5 follow-up (search-found, charged): the 2018+ QE+1 session (+18.7bp, t 2.8 in
kA_c5b's 15-cell session x era walk). Is it the rebound of a QE-0 fix dip
(sell at the fix, rebound next session), and what is its record?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa

if __name__ == "__main__":
    dx, uup, spy = own_series("DX-Y.NYB"), own_series("UUP"), own_series("SPY")
    me = month_ends(spy.index)
    for nm, c in (("DX", dx), ("UUP", uup)):
        a = pd.DatetimeIndex([c.index[c.index.searchsorted(d, side="right") - 1] for d in me if d >= c.index[0]])
        mp = pd.Series([d for d in me if d >= c.index[0]], index=a)
        s0 = fwd(c, a, 1, start=-1); s0.index = mp.reindex(s0.index).values
        s1 = fwd(c, a, 1); s1.index = mp.reindex(s1.index).values
        rows = []
        for lo in ("2008", "2013", "2018"):
            for lab, s in (("QE-0 session", s0), ("QE+1 session", s1)):
                q = s[s.index.month.isin([3, 6, 9, 12]) & (s.index >= lo)]
                o = s[~s.index.month.isin([3, 6, 9, 12]) & (s.index >= lo)]
                r = rec(q, f"{nm} {lab} {lo}+")
                r["ordME_pct"] = round(100 * o.mean(), 3)
                rows.append(r)
        show(rows, f"{nm} fix-dip / rebound check")
        q1 = s1[s1.index.month.isin([3, 6, 9, 12]) & (s1.index >= "2018")]
        q0 = s0.reindex(q1.index)
        print(f"  2018+ corr(QE-0, QE+1) = {q0.corr(q1):+.2f}; QE+1 after a down QE-0: "
              f"{100*q1[q0 < 0].mean():+.3f}% (n {int((q0<0).sum())}), after up QE-0: {100*q1[q0 >= 0].mean():+.3f}%")
        print("  2018+ QE+1:", ", ".join(f"{d.strftime('%y-%m')}:{1e4*v:+.0f}" for d, v in q1.items()))
