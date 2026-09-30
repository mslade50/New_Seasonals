"""C8 round 1: SHORT EEM / LONG beta*SPY from the quarter-end close, h=1..10,
h=5 pre-named. Beta = EX-ANTE trailing-252d daily OLS beta of EEM on SPY at the
anchor close (also fixed 0.93 as in W62). Controls: ordinary month-ends, all
days (same trailing-beta pair). Pre-specified dose: the QE-5->QE pair run-in
(long EEM - beta SPY) should predict the post-QE give-back (positive slope of
SHORT-pair return on run-in). September subset and 2013+ reported separately."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa
from kA_c5_dx_post_qe import regress  # noqa

HS = (1, 2, 3, 5, 7, 10)


def trailing_beta(y: pd.Series, x: pd.Series, n: int = 252) -> pd.Series:
    ry, rx = y.pct_change(), x.pct_change()
    cov = ry.rolling(n).cov(rx)
    var = rx.rolling(n).var()
    return cov / var


def pair_fwd(eem, spy, beta, anchors, h, start=0, sign=-1.0):
    """sign=-1: SHORT EEM, LONG beta SPY. Beta read at the ENTRY close."""
    fe, fs = fwd(eem, anchors, h, start), fwd(spy, anchors, h, start)
    b = beta.reindex(fe.index)
    return sign * (fe - b * fs)


if __name__ == "__main__":
    px = close_panel(["SPY", "EEM"]).dropna()
    spy, eem = px["SPY"], px["EEM"]
    beta = trailing_beta(eem, spy).dropna()
    fixed = pd.Series(0.93, index=px.index)
    print(f"ex-ante 252d beta: live (09-29) {beta.iloc[-1]:.3f}; QE-anchor median "
          f"{beta.reindex(month_ends(px.index)).median():.3f}")
    me = month_ends(px.index)
    me = me[me >= beta.index[0]]
    qe = me[me.month.isin([3, 6, 9, 12])]
    ome = me[~me.month.isin([3, 6, 9, 12])]
    print(f"{len(qe)} quarter-ends {qe[0].date()}..{qe[-1].date()}, {len(ome)} ordinary ME")

    for bl, B in (("trailing-252 beta", beta), ("fixed 0.93", fixed)):
        rows = []
        for h in HS:
            q = pair_fwd(eem, spy, B, qe, h)
            o = pair_fwd(eem, spy, B, ome, h)
            allw = -((eem.shift(-h) / eem - 1) - B.reindex(px.index) * (spy.shift(-h) / spy - 1)).dropna()
            allw = allw[allw.index >= beta.index[0]]
            rows += [rec(q, f"h={h} POST-QE short pair"), rec(o, f"h={h} ordinary ME"), rec(allw, f"h={h} all days")]
            d1, t1 = welch(q, o); d2, t2 = welch(q, allw)
            rows.append({"label": f"   QE-ordME {d1:+.3f}pp t{t1:+.2f} | QE-alldays {d2:+.3f}pp t{t2:+.2f}"})
        show(rows, f"C8 short EEM vs {bl}")

    B = beta
    for h in (3, 5, 10):
        q = pair_fwd(eem, spy, B, qe, h)
        show(eras(q, f"h={h}", ("2013-01-01", "2018-01-01"))
             + [rec(q[q.index.month == m], f"h={h} month {m}") for m in (3, 6, 9, 12)]
             + [rec(q[(q.index.month == 9) & (q.index >= "2013-01-01")], f"h={h} Sep 2013+"),
                rec(q[q.index.year % 4 == 2], f"h={h} midterm"), rec(q[q.index.year % 4 != 2], f"h={h} non-mid")],
             f"h={h} era / quarter / cycle")
        print("  concentration:", cluster_note(q.index, q.values))
        # legs
        fe = -fwd(eem, qe, h); fs = fwd(spy, qe, h)
        print(f"  legs h={h}: short EEM {100*fe.mean():+.3f}%, long beta*SPY {100*(B.reindex(fs.index)*fs).mean():+.3f}%")

    # dose: run-in (long pair QE-5 -> QE, beta at QE-5 entry) -> post-QE short pair
    run = -pair_fwd(eem, spy, B, qe, 5, start=-5)  # long-pair run-in
    for h in (1, 2, 3, 5, 10):
        q = pair_fwd(eem, spy, B, qe, h)
        print(f"\nDOSE h={h}: run-in -> post-QE SHORT pair: {regress(run.reindex(q.index), q)}")
        ok = run.reindex(q.index).notna()
        rr = run.reindex(q.index)[ok]
        t3 = pd.qcut(rr, 3, labels=["run-in low", "mid", "run-in high"])
        show([rec(q[ok][t3 == l], f"h={h} {l}") for l in ["run-in low", "mid", "run-in high"]], "dose terciles")
    print("\nrun-in itself (long pair QE-5->QE):", rec(run, "run-in"))
    o_run = -pair_fwd(eem, spy, B, ome, 5, start=-5)
    print("ordinary ME run-in:", rec(o_run, "oME run-in"))
    live_run = -((eem.iloc[-1] / eem.iloc[-5] - 1) - beta.iloc[-5] * (spy.iloc[-1] / spy.iloc[-5] - 1))
    print(f"live run-in so far (QE-5 close 09-23 -> QE-1 09-29, 4 sessions): {100*live_run:+.3f}%")
