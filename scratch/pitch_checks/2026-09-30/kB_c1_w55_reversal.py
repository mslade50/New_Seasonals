"""C1 round 1: W55 CHECK. Long the quarter's two worst nine-SPDRs by 63d, short
the two best, from the QE close to QE+5 (calendar anchor: entry = the QE
session's own close, exit QE+5 close). Rank convention: raw 63-session return
(ordering is what matters; not the trailing-252 percentile). Ranking known at
QE-1 close (live form: the 09-29 close); parent ranked at QE-10.
Positive number = the reversal pair earns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line, spread_at  # noqa

H = 5
idx = nyse_index()
px = close_panel(SPDR9 + ["SPY", "USO"]).reindex(idx)
P = px[SPDR9].values
r63 = pd.DataFrame({t: px[t] / px[t].shift(63) - 1 for t in SPDR9}).values
T = me_table(idx)


def build(rank_off: int, h: int = H, uni=None, k: int = 2):
    cols = list(range(len(SPDR9))) if uni is None else [SPDR9.index(u) for u in uni]
    out = []
    for _, a in T.iterrows():
        m = int(a.me_pos)
        s = m + rank_off
        if s < 63 or m + h >= len(idx):
            out.append((np.nan, np.nan, np.nan))
            continue
        out.append(spread_at(P[:, cols], r63[s, cols], m, m + h, k))
    arr = np.array(out)
    return arr[:, 0], arr[:, 1], arr[:, 2]


T["rev"], T["lo_leg"], T["hi_leg"] = build(-1)
T["rev_p10"], _, _ = build(-10)
T["rev_p0"], _, _ = build(0)
spy5 = (px["SPY"].shift(-H) / px["SPY"] - 1).values
T["spy"] = [spy5[int(m)] for m in T.me_pos]
A = T.dropna(subset=["rev"])
q, nq = A[A.qe], A[~A.qe]

rows = [stats_line(q.rev, q.me_date, "QE close->QE+5, rank@QE-1 (live form)"),
        stats_line(q.rev_p10, q.me_date, "  same, rank@QE-10 (parent W55 row)"),
        stats_line(q.rev_p0, q.me_date, "  same, rank@QE close (lag-0 look)"),
        stats_line(nq.rev, nq.me_date, "CTRL non-QE ME->ME+5 (tdom-matched)"),
        stats_line(q[q.year < 2018].rev, q[q.year < 2018].me_date, "QE pre-2018"),
        stats_line(q[q.year >= 2018].rev, q[q.year >= 2018].me_date, "QE 2018+  (W55 gate: > +0.40%)"),
        stats_line(nq[nq.year >= 2018].rev, nq[nq.year >= 2018].me_date, "non-QE 2018+"),
        stats_line(q[q.month == 9].rev, q[q.month == 9].me_date, "September QE (W55: not wrong-signed)"),
        stats_line(q[(q.month == 9) & (q.year >= 2018)].rev, q[(q.month == 9) & (q.year >= 2018)].me_date, "September QE 2018+"),
        stats_line(q[q.midterm].rev, q[q.midterm].me_date, "QE midterm years"),
        stats_line(q[(q.month == 9) & q.midterm].rev, q[(q.month == 9) & q.midterm].me_date, "Sep midterm")]
show(rows, "C1 reversal pair (long bottom-2, short top-2), QE close -> QE+5")

# all-days own drift of the same construction (rank at d-1, hold d->d+5)
alld = []
for p in range(64, len(idx) - H, 1):
    alld.append(spread_at(P, r63[p - 1], p, p + H)[0])
alld = np.array(alld)
print(f"\nCTRL own drift, every session (rank d-1, hold 5): mean {100*np.nanmean(alld):+.3f}% "
      f"n={np.isfinite(alld).sum()} hit {100*np.nanmean(alld[np.isfinite(alld)] > 0):.1f}%")
d = q.rev.mean() - nq.rev.mean()
se = np.sqrt(q.rev.var() / len(q) + nq.rev.var() / len(nq))
print(f"QE minus non-QE: {100*d:+.3f}pp  welch t {d/se:+.2f}")
w = int((q.rev > 0).sum())
print(f"QE record {w}-{len(q)-w}, sign p {sign_test(w, len(q)):.4f}; walk charge x4 "
      f"(parent's reversal table had 4 rows) -> {min(1, 4*sign_test(w, len(q))):.4f}")
print("concentration:", cluster_note(pd.DatetimeIndex(q.me_date), q.rev.values))
print(f"worst QE window {100*q.rev.min():.2f}% on {q.me_date[q.rev.idxmin()].date()}")
print(f"legs QE: long losers {100*q.lo_leg.mean():+.3f}%, short winners give "
      f"{-100*q.hi_leg.mean():+.3f}%, SPY same window {100*q.spy.mean():+.3f}%")
print(f"cost: 8 bp per unit spread; QE mean {1e4*q.rev.mean():.1f} bp -> {q.rev.mean()*1e4/8:.1f}x; "
      f"2018+ {1e4*q[q.year>=2018].rev.mean():.1f} bp -> {q[q.year>=2018].rev.mean()*1e4/8:.1f}x")

# energy leg: universe ex-XLE; and quarters where XLE sits in the top 2
T["rev_xe"], _, _ = build(-1, uni=[u for u in SPDR9 if u != "XLE"])
A = T.dropna(subset=["rev"])
q = A[A.qe]
xle_top = []
for _, a in q.iterrows():
    rr = r63[int(a.me_pos) - 1]
    xle_top.append(SPDR9.index("XLE") in np.argsort(rr)[-2:])
q = q.assign(xle_top=xle_top)
show([stats_line(q.rev_xe, q.me_date, "QE ex-XLE universe (8 SPDRs)"),
      stats_line(q[q.year >= 2018].rev_xe, q[q.year >= 2018].me_date, "QE ex-XLE 2018+"),
      stats_line(q[q.xle_top].rev, q[q.xle_top].me_date, "QE with XLE in top-2"),
      stats_line(q[~q.xle_top].rev, q[~q.xle_top].me_date, "QE with XLE not top-2")],
     "energy-leg decomposition")

# era table
q2 = q.assign(era=pd.cut(q.year, [1998, 2007, 2012, 2017, 2021, 2027]))
print("\nby era (QE mean %, n):")
print(q2.groupby("era", observed=True).rev.agg(lambda x: round(100 * x.mean(), 3)).to_string(),
      q2.groupby("era", observed=True).rev.size().to_string())
s9 = q[q.month == 9]
print("\nSeptember QE episodes:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s9.me_date, s9.rev)))
print("QE 2018+ episodes:", ", ".join(f"{d.strftime('%y-%m')}:{100*r:+.2f}" for d, r in
                                      zip(q[q.year >= 2018].me_date, q[q.year >= 2018].rev)))

# live legs
print("\nLIVE 63d returns at 2026-09-29 close (rank for the 09-30 MOC):")
print((100 * pd.Series(r63[-1], index=SPDR9)).sort_values(ascending=False).round(2).to_string())
u = px["USO"]
print(f"USO 09-29 1d {100*(u.iloc[-1]/u.iloc[-2]-1):+.2f}%, XLE 1d {100*(px['XLE'].iloc[-1]/px['XLE'].iloc[-2]-1):+.2f}%")
