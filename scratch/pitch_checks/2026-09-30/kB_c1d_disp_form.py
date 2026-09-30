"""C1 round 2c: the dispersion-conditioned form found in kB_c1b (checker's walk:
2 thresholds, p50/p80 of the QE top2-bottom2 63d gap). Threshold ladder,
by-session decomposition, ex-XLE universe (8-SPDR gap), September/midterm,
QE-vs-all-days contrast by era at matched dispersion. Live gap 22.4pp."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line  # noqa

idx = nyse_index()
px = close_panel(SPDR9).reindex(idx)
n = len(idx)
T = me_table(idx)


def engine(uni):
    P = px[uni].values
    R = pd.DataFrame({t: px[t] / px[t].shift(63) - 1 for t in uni}).values
    rev = {h: np.full(n, np.nan) for h in range(0, 11)}
    gap = np.full(n, np.nan)
    for p in range(64, n):
        rr = R[p - 1]
        ok = np.isfinite(rr) & np.isfinite(P[p])
        if ok.sum() < 5:
            continue
        o = np.argsort(rr[ok])
        lo, hi = np.flatnonzero(ok)[o[:2]], np.flatnonzero(ok)[o[-2:]]
        gap[p] = rr[hi].mean() - rr[lo].mean()
        for h in range(0, 11):
            if p + h < n:
                f = P[p + h] / P[p] - 1
                rev[h][p] = f[lo].mean() - f[hi].mean()
    return rev, gap, R


rev, gap, R = engine(SPDR9)
qe = T[T.qe].me_pos.values
qe = qe[np.isfinite(rev[5][qe])]
alld = np.arange(64, n - 5)
print(f"LIVE gap {100*gap[-1]:.1f}pp (rank@09-29 applies to the 09-30 anchor; gap[-1] uses 09-26 rank row, "
      f"recomputed on the last row: {100*(np.sort(R[-1])[-2:].mean()-np.sort(R[-1])[:2].mean()):.1f}pp)")
live_gap = np.sort(R[-1])[-2:].mean() - np.sort(R[-1])[:2].mean()

rows = []
for pc in (0, 50, 60, 70, 75, 80, 85, 90):
    thr = np.quantile(gap[qe], pc / 100) if pc else -1
    s = qe[gap[qe] >= thr]
    a = stats_line(rev[5][s], idx[s], f"QE gap>=p{pc} ({100*max(thr,0):.1f}pp)")
    ad = alld[gap[alld] >= thr]
    ep = declusters(idx[ad], 5, idx)
    pos = pd.Series(range(n), index=idx)[ep].values
    a["alldays_ep"] = 100 * np.nanmean(rev[5][pos])
    a["QE_18+"] = 100 * np.nanmean(rev[5][s[idx[s].year >= 2018]])
    a["alld_18+"] = 100 * np.nanmean(rev[5][pos[idx[pos].year >= 2018]])
    a["live_in"] = live_gap >= thr
    rows.append(a)
show(rows, "threshold ladder (QE h=5 vs all-days declustered at the same gap)")

thr50 = np.quantile(gap[qe], 0.5)
thr80 = np.quantile(gap[qe], 0.8)
for lbl, thr in (("p50", thr50), ("p80", thr80)):
    s = qe[gap[qe] >= thr]
    rows = []
    for d in range(1, 6):
        inc = rev[d][s] - (rev[d - 1][s] if d > 1 else 0)
        # session increment of a cumulative pair return (approximate, equal-weight rebal-free)
        r = stats_line(inc, idx[s], f"QE+{d}")
        rows.append(r)
    show(rows, f"by-session increments, QE gap>={lbl} (N={len(s)}); mechanism names QE+1")
    y = idx[s].year
    show([stats_line(rev[5][s][idx[s].month == 9], idx[s][idx[s].month == 9], "September"),
          stats_line(rev[5][s][y % 4 == 2], idx[s][y % 4 == 2], "midterm"),
          stats_line(rev[5][s][y < 2018], idx[s][y < 2018], "pre-2018"),
          stats_line(rev[5][s][y >= 2018], idx[s][y >= 2018], "2018+")], f"splits, gap>={lbl}")
    print(f"  concentration gap>={lbl}:", cluster_note(idx[s], rev[5][s]))

# ex-XLE universe with its own gap
rev8, gap8, R8 = engine([t for t in SPDR9 if t != "XLE"])
qe8 = qe[np.isfinite(rev8[5][qe])]
lg8 = np.sort(R8[-1])[-2:].mean() - np.sort(R8[-1])[:2].mean()
rows = []
for pc in (0, 50, 80):
    thr = np.quantile(gap8[qe8], pc / 100) if pc else -1
    s = qe8[gap8[qe8] >= thr]
    r = stats_line(rev8[5][s], idx[s], f"ex-XLE QE gap8>=p{pc}")
    r["2018+"] = 100 * np.nanmean(rev8[5][s[idx[s].year >= 2018]])
    r["live_in"] = lg8 >= thr
    rows.append(r)
show(rows, f"ex-XLE universe (live 8-SPDR gap {100*lg8:.1f}pp; live legs would be long XLU/XLI short XLV/XLK)")
