"""C7 round 1: long the REIT ETF against ex-ante beta-SPY from the quarter-end
close (calendar anchor, entry = QE session's own close), h=1..10, LONG.
Vehicles: IYR (2000+, primary history), VNQ (2004+), XLRE (2015+, modern).
Primary gate (pre-specified): REIT r63 trailing-252 percentile (pitch_lab.pct_rank,
tape convention) <= 5 at the QE close; conditioner: ^TNX 63d-change pct >= 90.
Beta: trailing-252 daily OLS on SPY known at the QE-1 close."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import SPDR9, nyse_index, me_table, stats_line, spread_at, exante_beta  # noqa

idx = nyse_index()
px = close_panel(["IYR", "VNQ", "XLRE", "SPY", "^TNX"] + SPDR9).reindex(idx)
T = me_table(idx)
tnx = px["^TNX"]
tnx63 = rolling_on_valid(tnx.dropna().diff(63).reindex(idx),
                         lambda x: x.rolling(252).rank(pct=True) * 100)


def resid_path(reit: str, h: int):
    b = exante_beta(px[reit], px["SPY"]).shift(1)
    r = px[reit].shift(-h) / px[reit] - 1
    s = px["SPY"].shift(-h) / px["SPY"] - 1
    return r - b * s, r, b


def qe_frame(reit: str, h: int):
    res, raw, b = resid_path(reit, h)
    rk = pct_rank(px[reit], 63)
    rows = []
    for _, a in T.iterrows():
        m = int(a.me_pos)
        rows.append({"d": a.me_date, "qe": a.qe, "year": a.year, "month": a.month,
                     "mid": a.midterm, "res": res.iloc[m], "raw": raw.iloc[m],
                     "rk": rk.iloc[m], "rk_prev": rk.iloc[m - 1], "tnx": tnx63.iloc[m],
                     "beta": b.iloc[m]})
    return pd.DataFrame(rows).dropna(subset=["res", "rk"]), res, rk


for reit in ("IYR", "VNQ", "XLRE"):
    F, res, rk = qe_frame(reit, 5)
    q, nq = F[F.qe], F[~F.qe]
    g = q[q.rk <= 5]
    gt = q[q.tnx >= 90]
    gj = q[(q.rk <= 5) & (q.tnx >= 90)]
    st = idx[(rk <= 5).values & res.notna().values]
    ep = declusters(st, 5, idx)
    rows = [stats_line(q.res, q.d, f"{reit} QE ungated resid h=5"),
            stats_line(nq.res, nq.d, "CTRL non-QE month-ends resid"),
            stats_line(res.dropna().values, res.dropna().index, "CTRL all days resid (own drift)"),
            stats_line(g.res, g.d, "QE & r63<=5 (PRIMARY)"),
            stats_line(q[q.rk_prev <= 5].res, q[q.rk_prev <= 5].d, "QE & r63<=5 @QE-1 (known)"),
            stats_line(q[q.rk <= 10].res, q[q.rk <= 10].d, "QE & r63<=10"),
            stats_line(q[q.rk <= 20].res, q[q.rk <= 20].d, "QE & r63<=20"),
            stats_line(gt.res, gt.d, "QE & TNX63>=90"),
            stats_line(gj.res, gj.d, "QE & r63<=5 & TNX63>=90"),
            stats_line(nq[nq.rk <= 5].res, nq[nq.rk <= 5].d, "non-QE ME & r63<=5"),
            stats_line(res.loc[ep].values, ep, "CTRL r63<=5 ALL days (episodes)"),
            stats_line(g.raw, g.d, "  PRIMARY raw (unhedged) return")]
    show(rows, f"C7 {reit} vs beta-SPY from the QE close, h=5")
    if reit == "IYR":
        print("PRIMARY episodes:", ", ".join(f"{d.date()}:{100*r:+.2f}(rk{k:.1f},tnx{t:.0f})"
                                             for d, r, k, t in zip(g.d, g.res, g.rk, g.tnx)))
        print("TNX-only episodes:", ", ".join(f"{d.date()}:{100*r:+.2f}(rk{k:.0f})"
                                              for d, r, k in zip(gt.d, gt.res, gt.rk)))
        show(era_split(pd.DatetimeIndex(q.d), q.res.values), "IYR QE ungated era split")
        show(era_split(pd.DatetimeIndex(g.d), g.res.values), "IYR PRIMARY era split")

# horizon view of the primary on IYR
hz = []
for h in (1, 2, 3, 5, 7, 10):
    F, _, _ = qe_frame("IYR", h)
    q = F[F.qe]
    hz.append(stats_line(q[q.rk <= 5].res, q[q.rk <= 5].d, f"IYR PRIMARY h={h}"))
    hz.append(stats_line(q.res, q.d, f"   IYR QE ungated h={h}"))
show(hz, "IYR horizon rows")

# overlap with C1: C1 pair (rank@QE-1, QE->QE+5) vs C7 IYR resid h=5 on shared QEs
P = px[SPDR9].values
r63 = pd.DataFrame({t: px[t] / px[t].shift(63) - 1 for t in SPDR9}).values
F, _, _ = qe_frame("IYR", 5)
q = F[F.qe].copy()
pos = pd.Series(range(len(idx)), index=idx)
q["c1"] = [spread_at(P, r63[pos[d] - 1], pos[d], pos[d] + 5)[0] for d in q.d]
ok = q.dropna(subset=["c1"])
print(f"\ncorr(C7 IYR resid h=5, C1 pair h=5) over {len(ok)} shared QEs: {ok[['res','c1']].corr().iloc[0,1]:+.3f}; "
      f"primary-gated QEs C1 pair mean {100*ok[ok.rk<=5].c1.mean():+.3f}% (n={int((ok.rk<=5).sum())})")

for reit in ("IYR", "VNQ", "XLRE"):
    rk = pct_rank(px[reit], 63)
    b = exante_beta(px[reit], px["SPY"])
    print(f"LIVE {reit}: r63 pct {rk.iloc[-1]:.1f}, beta {b.iloc[-1]:.2f}, 63d {100*(px[reit].iloc[-1]/px[reit].iloc[-64]-1):+.2f}%")
print(f"LIVE TNX 63d-change pct {tnx63.iloc[-1]:.1f}")
