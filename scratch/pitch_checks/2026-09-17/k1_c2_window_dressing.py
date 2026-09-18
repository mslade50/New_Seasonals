"""c2 round 1: quarter-end window dressing, top2-minus-bottom2 63d SPDRs,
QE-9 -> QE (signal = QE-10 close). Controls: non-quarter month-ends, all days.
Mechanism signature: spread reverses QE -> QE+5. Single-stock decile version."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

H = 9
idx = nyse_index()
px = close_panel(SPDR9 + ["SPY"]).reindex(idx)
P = px[SPDR9]
r63 = pd.DataFrame({t: P[t] / P[t].shift(63) - 1 for t in SPDR9})


def spread_series(prices, rank, k, h, lag):
    """Signal-date aligned: top-k minus bottom-k (by rank at signal) forward
    return entering close s+lag, exiting s+lag+h."""
    fwd = prices.shift(-(lag + h)) / prices.shift(-lag) - 1
    R = rank.values
    F = fwd.values
    out = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        rr, ff = R[i], F[i]
        ok = ~np.isnan(rr) & ~np.isnan(ff)
        if ok.sum() < 2 * k + 1:
            continue
        rr2, ff2 = rr[ok], ff[ok]
        o = np.argsort(rr2)
        out[i] = ff2[o[-k:]].mean() - ff2[o[:k]].mean()
    return pd.Series(out, index=prices.index)


sp = spread_series(P, r63, 2, H, 1)
A = anchors(idx, -10)
A["ret"] = sp.values[A.sig_pos]
A = A.dropna(subset=["ret"])

rows = [stats_line(A[A.qe].ret, A[A.qe].sig_date, "QE-9->QE all quarters"),
        stats_line(A[~A.qe].ret, A[~A.qe].sig_date, "non-QE ME-9->ME"),
        stats_line(A[A.month == 9].ret, A[A.month == 9].sig_date, "September QE"),
        stats_line(A[A.qe & (A.month != 9)].ret, A[A.qe & (A.month != 9)].sig_date, "Mar/Jun/Dec QE"),
        stats_line(A[A.month == 12].ret, A[A.month == 12].sig_date, "December QE"),
        stats_line(A[A.qe & A.midterm].ret, A[A.qe & A.midterm].sig_date, "QE midterm"),
        stats_line(A[(A.month == 9) & A.midterm].ret, A[(A.month == 9) & A.midterm].sig_date, "Sep midterm"),
        stats_line(A[(A.month == 9) & (A.year < 2018)].ret, A[(A.month == 9) & (A.year < 2018)].sig_date, "Sep pre-2018"),
        stats_line(A[(A.month == 9) & (A.year >= 2018)].ret, A[(A.month == 9) & (A.year >= 2018)].sig_date, "Sep 2018+"),
        stats_line(sp.dropna().values, sp.dropna().index, "ALL days (9-session windows)")]
show(rows, "c2 SPDR top2-bottom2 63d spread, h=9 lag=1")

q = A[A.qe]
nq = A[~A.qe]
print(f"\nQE minus non-QE month-end: {100*(q.ret.mean()-nq.ret.mean()):+.3f}pp; "
      f"welch t {(q.ret.mean()-nq.ret.mean())/np.sqrt(q.ret.var()/len(q)+nq.ret.var()/len(nq)):+.2f}")
print("QE era:")
show([stats_line(q[q.year < 2013].ret, q[q.year < 2013].sig_date, "QE pre-2013"),
      stats_line(q[(q.year >= 2013) & (q.year < 2018)].ret, q[(q.year >= 2013) & (q.year < 2018)].sig_date, "QE 2013-17"),
      stats_line(q[q.year >= 2018].ret, q[q.year >= 2018].sig_date, "QE 2018+"),
      stats_line(nq[nq.year >= 2018].ret, nq[nq.year >= 2018].sig_date, "non-QE 2018+")])
s9 = A[A.month == 9]
print("\nSeptember episodes:", ", ".join(f"{d.year}:{100*r:+.2f}" for d, r in zip(s9.me_date, s9.ret)))
print("concentration QE:", cluster_note(pd.DatetimeIndex(q.sig_date), q.ret.values))
print(f"worst QE window {100*q.ret.min():.2f}% on {q.me_date[q.ret.idxmin()].date()}")

# by month of year
bym = A.groupby("month").ret.agg(["mean", "count", lambda x: (x > 0).mean()])
bym["mean"] *= 100
print("\nby month (mean %, n, hit):\n", bym.round(3).to_string())

# mechanism: reversal QE -> QE+5 using the QE-10 ranking
rev = []
Pv = P.values
for _, a in A.iterrows():
    m, s = int(a.me_pos), int(a.sig_pos)
    if m + 5 >= len(idx):
        rev.append(np.nan)
        continue
    rr = r63.values[s]
    f = Pv[m + 5] / Pv[m] - 1
    ok = ~np.isnan(rr) & ~np.isnan(f)
    o = np.argsort(rr[ok])
    rev.append(f[ok][o[-2:]].mean() - f[ok][o[:2]].mean())
A["rev"] = rev
qq, nn = A[A.qe], A[~A.qe]
show([stats_line(qq.rev, qq.sig_date, "QE -> QE+5 (reversal predicts <0)"),
      stats_line(nn.rev, nn.sig_date, "non-QE ME -> ME+5"),
      stats_line(A[A.month == 9].rev, A[A.month == 9].sig_date, "Sep QE -> +5"),
      stats_line(A[A.month == 10].rev, A[A.month == 10].sig_date, "Oct ME -> +5 (fund FYE)")],
     "mechanism: post-quarter reversal")
print(f"corr(run-in spread, post-QE spread) at QE: {qq[['ret','rev']].corr().iloc[0,1]:+.3f}")

# costs: 4 legs at 0.5 weight each = 2x notional; ~4bp RT per leg notional
print(f"\ncost: ~8 bp per unit spread (2x notional x 4bp). QE mean "
      f"{1e4*q.ret.mean():.1f} bp -> {1e4*q.ret.mean()/8:.1f}x; excess over non-QE "
      f"{1e4*(q.ret.mean()-nq.ret.mean()):.1f} bp -> {1e4*(q.ret.mean()-nq.ret.mean())/8:.1f}x")

# live state
last = idx[-1]
print(f"\nLIVE {last.date()} 63d ranks:")
print((100 * r63.iloc[-1]).sort_values(ascending=False).round(2).to_string())
xtra = close_panel(["XLRE", "XLC"]).reindex(idx)
for t in ["XLRE", "XLC"]:
    print(f"  {t} 63d {100*(xtra[t].iloc[-1]/xtra[t].iloc[-64]-1):+.2f}% (note only)")
