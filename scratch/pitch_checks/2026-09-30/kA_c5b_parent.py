"""C5 round 2 (on the UNDOSED parent only, for the park decision; the dosed
candidate died in round 1): post-QE long DX by session and era, LOYO, drop-best,
QE vs ordinary ME by era, and whether the post-QE gain reverses the QE-5->QE
run-in (a reversal needs something to reverse)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa

if __name__ == "__main__":
    dx, spy = own_series("DX-Y.NYB"), own_series("SPY")
    me = month_ends(spy.index)
    a = pd.DatetimeIndex([dx.index[dx.index.searchsorted(d, side="right") - 1] for d in me])
    mp = pd.Series(me, index=a)
    base = dx.pct_change().dropna()
    rows = []
    for lo, hi in (("2000", "2008"), ("2008", "2018"), ("2018", "2027")):
        b = base[(base.index >= lo) & (base.index < hi)].mean()
        row = {"era": f"{lo}-{int(hi)-1}", "base_bp": round(1e4 * b, 2)}
        for k in range(1, 6):
            f = fwd(dx, a, 1, start=k - 1); f.index = mp.reindex(f.index).values
            q = f[f.index.month.isin([3, 6, 9, 12]) & (f.index >= lo) & (f.index < hi)]
            o = f[~f.index.month.isin([3, 6, 9, 12]) & (f.index >= lo) & (f.index < hi)]
            row[f"QE+{k}"] = f"{1e4*(q.mean()-b):+.1f}(t{(q.mean()-b)/(q.std()/np.sqrt(len(q))):+.1f})"
            row[f"oME+{k}"] = f"{1e4*(o.mean()-b):+.1f}"
        rows.append(row)
    print("=== excess LONG DX bp per post-QE session vs era all-days; oME = ordinary month-end ===")
    print(pd.DataFrame(rows).to_string(index=False))

    for h in (2, 5):
        f = fwd(dx, a, h); f.index = mp.reindex(f.index).values
        q = f[f.index.month.isin([3, 6, 9, 12])]
        o = f[~f.index.month.isin([3, 6, 9, 12])]
        out = []
        for lo, hi in (("2000", "2008"), ("2008", "2018"), ("2018", "2027")):
            qq = q[(q.index >= lo) & (q.index < hi)]
            oo = o[(o.index >= lo) & (o.index < hi)]
            d, t = welch(qq, oo)
            r = rec(qq, f"h={h} QE {lo}-{int(hi)-1}")
            r["minus_ordME_pp"] = round(d, 3); r["t_vs_ordME"] = round(t, 2)
            out.append(r)
        show(out, f"h={h} QE vs ordinary ME by era")
        loyo = [100 * q[q.index.year != y].mean() for y in sorted(set(q.index.year))]
        srt = q.sort_values(ascending=False)
        print(f"  h={h} LOYO min {min(loyo):+.3f}  drop-best-3 {100*srt.iloc[3:].mean():+.3f}%  "
              f"years positive {(q.groupby(q.index.year).sum() > 0).sum()} of {q.index.year.nunique()}")
        run = fwd(dx, a, 5, start=-5); run.index = mp.reindex(run.index).values
        rq = run.reindex(q.index)
        ok = rq.notna()
        b = np.polyfit(rq[ok], q[ok], 1)[0]
        print(f"  h={h} run-in (QE-5->QE) -> post-QE slope {b:+.4f}  spearman {rq[ok].corr(q[ok], method='spearman'):+.3f}")
        t3 = pd.qcut(rq[ok], 3, labels=["DX fell in", "mid", "DX rose in"])
        show([rec(q[ok][t3 == l], f"h={h} run-in {l}") for l in ["DX fell in", "mid", "DX rose in"]], "run-in terciles")
