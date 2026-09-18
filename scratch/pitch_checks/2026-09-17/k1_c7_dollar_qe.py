"""c7 round 1: long the dollar (DX-Y.NYB) QE-9 -> QE vs non-QE ME-9 -> ME,
September, pre/post 2008 and 2015 (Basel III leverage-ratio quarter-end
window dressing), the 5d-rank >= 95 row, UUP as a row, per-session
decomposition and post-QE reversal."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

idx = nyse_index()
raw = close_panel(["DX-Y.NYB", "UUP"])
dx_own = raw["DX-Y.NYB"].dropna()
r5_own = pct_rank(dx_own, 5)
px = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
r5 = r5_own.reindex(idx.union(r5_own.index)).ffill().reindex(idx)
A = anchors(idx, -10)
A = A[A.me_pos + 5 < len(idx)].copy()
D, U = px["DX-Y.NYB"].values, px["UUP"].values


def win(v, a, b):
    return [v[m + b] / v[m + a] - 1 for m in A.me_pos]


A["dx"] = win(D, -9, 0)
A["uup"] = win(U, -9, 0)
A["post5"] = win(D, 0, 5)
A["r5"] = r5.values[A.sig_pos]
A = A.dropna(subset=["dx"])
q, nq = A[A.qe], A[~A.qe]

allw = pd.Series(D).shift(-10) / pd.Series(D).shift(-1) - 1
alld = pd.Series(allw.values, index=idx)


def rowset(col, frame_q, frame_nq, tag):
    return [stats_line(frame_q[col], frame_q.sig_date, f"{tag} QE"),
            stats_line(frame_nq[col], frame_nq.sig_date, f"{tag} non-QE ME")]


show(rowset("dx", q, nq, "all yrs")
     + [stats_line(A[A.month == 9].dx, A[A.month == 9].sig_date, "September QE"),
        stats_line(A[A.month == 12].dx, A[A.month == 12].sig_date, "December QE"),
        stats_line(A[A.dx.notna()].dx, A.sig_date, "all month-ends"),
        stats_line(alld.dropna().values, alld.dropna().index, "all days lag1 h9")],
     "DXY QE-9 -> QE")
print(f"QE minus non-QE: {100*(q.dx.mean()-nq.dx.mean()):+.3f}pp welch t "
      f"{(q.dx.mean()-nq.dx.mean())/np.sqrt(q.dx.var()/len(q)+nq.dx.var()/len(nq)):+.2f}")

rows = []
for lo, hi, tag in [(1999, 2007, "2000-07"), (2008, 2099, "2008+"), (2008, 2014, "2008-14"),
                    (2015, 2099, "2015+"), (2018, 2099, "2018+")]:
    fq = q[(q.year >= lo) & (q.year <= hi)]
    fn = nq[(nq.year >= lo) & (nq.year <= hi)]
    rows += rowset("dx", fq, fn, tag)
    rows.append({"label": f"  {tag} QE-nonQE pp", "mean_pct": 100 * (fq.dx.mean() - fn.dx.mean())})
show(rows, "era split (the CIP funding story is post-2008)")
s9 = A[A.month == 9]
show([stats_line(s9[s9.year >= 2008].dx, s9[s9.year >= 2008].sig_date, "Sep 2008+"),
      stats_line(s9[s9.year < 2008].dx, s9[s9.year < 2008].sig_date, "Sep pre-2008"),
      stats_line(q[q.midterm].dx, q[q.midterm].sig_date, "QE midterm"),
      stats_line(s9[s9.midterm].dx, s9[s9.midterm].sig_date, "Sep midterm")], "September / midterm")
print("Sep:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s9.me_date, s9.dx)))

# gate row: 5d rank >= 95 at the signal close
g = A[A.r5 >= 95]
show([stats_line(g[g.qe].dx, g[g.qe].sig_date, "r5>=95 QE"),
      stats_line(g[~g.qe].dx, g[~g.qe].sig_date, "r5>=95 non-QE ME"),
      stats_line(A[(A.r5 < 95) & A.qe].dx, A[(A.r5 < 95) & A.qe].sig_date, "r5<95 QE (complement)")],
     "gate row: DXY 5d rank >= 95 at signal")
gm = (r5 >= 95) & alld.notna()
gd = idx[gm.values]
ep = declusters(gd, 9, idx)
show([stats_line(alld.loc[ep].values, ep, "r5>=95 ALL days, declustered 9td"),
      stats_line(alld.dropna().values, alld.dropna().index, "all days")], "gate at all dates")
print(f"LIVE DXY r5 at 2026-09-16: {r5.iloc[-1]:.1f}")

# UUP row (registry-dead on drag; a row, not a vehicle)
uq, un = q[q.uup.notna()], nq[nq.uup.notna()]
show(rowset("uup", uq, un, "UUP"), "UUP row")

# per-session decomposition, QE vs non-QE, 2008+
print("\nper-session mean bp (close k-1 -> k), sessions relative to ME:")
out = []
for k in range(-9, 6):
    rq = [D[m + k] / D[m + k - 1] - 1 for m in q[q.year >= 2008].me_pos]
    rn = [D[m + k] / D[m + k - 1] - 1 for m in nq[nq.year >= 2008].me_pos]
    rqa = [D[m + k] / D[m + k - 1] - 1 for m in q.me_pos]
    out.append({"k": k, "QE08+_bp": 1e4 * np.nanmean(rq), "QE08+_hit": 100 * np.nanmean(np.array(rq) > 0),
                "nonQE08+_bp": 1e4 * np.nanmean(rn), "QEall_bp": 1e4 * np.nanmean(rqa)})
show(out)
print(f"all-days mean session bp: {1e4*np.nanmean(D[1:]/D[:-1]-1):+.2f}")
show([stats_line(q.post5, q.sig_date, "QE -> QE+5"), stats_line(nq.post5, nq.sig_date, "non-QE ME -> +5"),
      stats_line(q[q.year >= 2008].post5, q[q.year >= 2008].sig_date, "QE -> +5, 2008+")],
     "post-QE reversal (funding story predicts <0)")
print("concentration QE:", cluster_note(pd.DatetimeIndex(q.sig_date), q.dx.values))
print(f"cost: DX futures ~1bp RT, UUP ~4bp. QE mean {1e4*q.dx.mean():.1f}bp; "
      f"QE-nonQE {1e4*(q.dx.mean()-nq.dx.mean()):.1f}bp")
