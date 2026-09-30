"""C1 round 2b: is the high-dispersion split a quarter-end object or generic
sector 63d reversal at high dispersion? Same pair (rank d-1, hold 5) at
non-QE month-ends and on all days, split by the top2-bottom2 63d gap
percentile (thresholds taken from the QE distribution). Plus: who is in the
legs of the top-quintile QE episodes (XLE share)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line  # noqa

idx = nyse_index()
px = close_panel(SPDR9).reindex(idx)
P = px[SPDR9].values
R = pd.DataFrame({t: px[t] / px[t].shift(63) - 1 for t in SPDR9}).values
H = 5
n = len(idx)
rev = np.full(n, np.nan)
gap = np.full(n, np.nan)
xle_in = np.zeros(n, bool)
for p in range(64, n - H):
    rr = R[p - 1]
    f = P[p + H] / P[p] - 1
    ok = np.isfinite(rr) & np.isfinite(f)
    if ok.sum() < 5:
        continue
    o = np.argsort(rr[ok])
    rev[p] = f[ok][o[:2]].mean() - f[ok][o[-2:]].mean()
    gap[p] = rr[ok][o[-2:]].mean() - rr[ok][o[:2]].mean()
    names = np.array(SPDR9)[ok]
    xle_in[p] = "XLE" in set(names[o[:2]]) | set(names[o[-2:]])
T = me_table(idx)
qe_pos = T[T.qe].me_pos.values
me_pos = T[~T.qe].me_pos.values
q80 = np.nanquantile(gap[qe_pos], 0.8)
q50 = np.nanquantile(gap[qe_pos], 0.5)
alld = np.arange(64, n - H)
ep_all = declusters(idx[alld[gap[alld] >= q80]], H, idx)
pos = pd.Series(range(n), index=idx)
ep_pos = pos[ep_all].values
rows = []
for lbl, ps in (("QE", qe_pos), ("non-QE ME", me_pos), ("ALL days", alld)):
    for tl, thr in (("gap>=QE-p80", q80), ("gap>=QE-p50", q50), ("gap<QE-p50", -1)):
        sel = ps[(gap[ps] >= thr) if thr > 0 else (gap[ps] < q50)]
        sel = sel[np.isfinite(rev[sel])]
        r = stats_line(rev[sel], idx[sel], f"{lbl} {tl}")
        r["2018+"] = 100 * np.nanmean(rev[sel][idx[sel].year >= 2018]) if len(sel) else np.nan
        r["XLE_in_legs%"] = 100 * xle_in[sel].mean() if len(sel) else np.nan
        rows.append(r)
show(rows, f"pair h=5 by dispersion (QE p80 {100*q80:.1f}pp, p50 {100*q50:.1f}pp)")
print(stats_line(rev[ep_pos], idx[ep_pos], "ALL days gap>=p80, declustered episodes"))
sel = qe_pos[(gap[qe_pos] >= q80) & np.isfinite(rev[qe_pos])]
print("QE top-quintile episodes:", ", ".join(f"{idx[p].date()}:{100*rev[p]:+.2f}{'*' if xle_in[p] else ''}" for p in sel),
      "(* = XLE in a leg)")
sel2 = sel[~xle_in[sel]]
print(stats_line(rev[sel2], idx[sel2], "QE top quintile, XLE NOT in a leg"))
print(f"LIVE gap {100*(np.sort(R[-1])[-2:].mean()-np.sort(R[-1])[:2].mean()):.1f}pp")
