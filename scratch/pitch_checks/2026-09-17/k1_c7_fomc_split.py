"""c7 confound check: every quarter-end month carries an SEP FOMC ~2 weeks
before QE. Split QE-9 -> QE DXY windows by FOMC decision inside the hold vs
at/before the signal close (today's configuration), and compare against a
post-FOMC control in non-QE months. Also December (year-end funding peak)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_common import *  # noqa

idx = nyse_index()
raw = close_panel(["DX-Y.NYB"])
px = raw.reindex(idx.union(raw.index)).ffill().reindex(idx)
D = px["DX-Y.NYB"].values
A = anchors(idx, -10)
A = A[A.me_pos < len(idx)].copy()
A["dx"] = [D[m] / D[m - 9] - 1 for m in A.me_pos]
fomc = pd.DatetimeIndex(load_events(["fomc_decision"])["date"])
fomc = fomc[fomc <= idx[-1]]


def fomc_in_hold(sig_pos):
    lo, hi = idx[sig_pos + 1], idx[sig_pos + 10]
    return bool(((fomc > lo) & (fomc <= hi)).any())


def fomc_near_before(sig_pos, back=10):
    lo, hi = idx[max(0, sig_pos - back)], idx[sig_pos + 1]
    return bool(((fomc >= lo) & (fomc <= hi)).any())


A["f_in"] = [fomc_in_hold(int(s)) for s in A.sig_pos]
A["f_before"] = [fomc_near_before(int(s)) for s in A.sig_pos]
A = A.dropna(subset=["dx"])
for era, fr in [("all", A), ("2008+", A[A.year >= 2008])]:
    q, nq = fr[fr.qe], fr[~fr.qe]
    show([stats_line(q[q.f_in].dx, q[q.f_in].sig_date, f"{era} QE, FOMC inside hold"),
          stats_line(q[~q.f_in].dx, q[~q.f_in].sig_date, f"{era} QE, no FOMC in hold"),
          stats_line(q[q.f_before & ~q.f_in].dx, q[q.f_before & ~q.f_in].sig_date,
                     f"{era} QE, FOMC in 10td before entry (LIVE config)"),
          stats_line(nq[nq.f_in].dx, nq[nq.f_in].sig_date, f"{era} non-QE ME, FOMC inside hold"),
          stats_line(nq[~nq.f_in].dx, nq[~nq.f_in].sig_date, f"{era} non-QE ME, no FOMC in hold"),
          stats_line(nq[nq.f_before & ~nq.f_in].dx, nq[nq.f_before & ~nq.f_in].sig_date,
                     f"{era} non-QE ME, FOMC in 10td before entry")],
         f"DXY QE-9->QE by FOMC placement ({era})")
    s9 = fr[fr.month == 9]
    show([stats_line(s9[s9.f_in].dx, s9[s9.f_in].sig_date, f"{era} Sep FOMC inside"),
          stats_line(s9[~s9.f_in].dx, s9[~s9.f_in].sig_date, f"{era} Sep FOMC not inside"),
          stats_line(fr[fr.month == 12].dx, fr[fr.month == 12].sig_date, f"{era} December (year-end)"),
          stats_line(fr[fr.month == 3].dx, fr[fr.month == 3].sig_date, f"{era} March"),
          stats_line(fr[fr.month == 6].dx, fr[fr.month == 6].sig_date, f"{era} June")],
         f"quarter detail ({era})")
s9 = A[(A.month == 9) & (A.year >= 2008)]
print("Sep 2008+ (year, ret%, FOMC in hold):",
      ", ".join(f"{d.year}:{100*r:+.2f}:{'F' if f else '-'}" for d, r, f in zip(s9.me_date, s9.dx, s9.f_in)))

# post-FOMC control: DXY from decision+1 close over 9 sessions, all decisions
pos = pd.Series(range(len(idx)), index=idx)
rows = []
for d in fomc:
    p = idx.searchsorted(d)
    if p + 10 >= len(idx) or p >= len(idx) or idx[p] != d:
        continue
    rows.append({"date": d, "ret": D[p + 10] / D[p + 1] - 1, "qm": d.month in (3, 6, 9, 12)})
F = pd.DataFrame(rows)
F = F[F.date >= "2008-01-01"]
show([stats_line(F[F.qm].ret, F[F.qm].date, "2008+ FOMC in QE month, dec+1 -> +10"),
      stats_line(F[~F.qm].ret, F[~F.qm].date, "2008+ FOMC in non-QE month, dec+1 -> +10"),
      stats_line(F[F.date.dt.month == 9].ret, F[F.date.dt.month == 9].date, "2008+ Sep FOMC dec+1 -> +10")],
     "post-FOMC dollar control (today's entry = decision+1)")
