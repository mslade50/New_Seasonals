"""C5 round 1: LONG DX-Y.NYB from the quarter-end close, h=1..10, dosed by the
quarter's SPY-minus-EFA 63d return at the QE close (FX-hedge rebalance
reversal). Pre-specified sign LONG, pre-specified dose SPY63-EFA63 (signed
regression + terciles). Controls: ordinary month-ends at the same h, all days,
UUP. Mechanism leg: does the dose predict dollar WEAKNESS into the QE close
(QE-5 -> QE)? If nothing is sold into the close, nothing reverses.
State check: DX within 0.5% of its 252 high / z10 >= 2 at the QE close."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa

HS = (1, 2, 3, 5, 7, 10)


def regress(x: pd.Series, y: pd.Series) -> str:
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    b = np.polyfit(x, y, 1)
    res = y - np.polyval(b, x)
    se = np.sqrt(res.var(ddof=2) / ((x - x.mean())**2).sum())
    r2 = 1 - res.var() / y.var()
    rho = x.corr(y, method="spearman")
    return f"slope {b[0]:+.4f} t {b[0]/se:+.2f} R2 {r2:.4f} spearman {rho:+.3f} n {len(x)}"


if __name__ == "__main__":
    dx, uup = own_series("DX-Y.NYB"), own_series("UUP")
    spy, efa, eem = own_series("SPY"), own_series("EFA"), own_series("EEM")
    me = month_ends(spy.index)  # US equity calendar month-ends (the hedge flow is on equity books)
    # map each ME to DX's own last bar at-or-before it
    def on_cal(c, anchors):
        return pd.DatetimeIndex([c.index[c.index.searchsorted(d, side="right") - 1] for d in anchors])
    qe_mask = me.month.isin([3, 6, 9, 12])
    dose = (spy / spy.shift(63) - 1) - (efa / efa.shift(63) - 1)
    dose_eem = (spy / spy.shift(63) - 1) - (eem / eem.shift(63) - 1)
    print("live dose (09-29):", round(100 * dose.iloc[-1], 2), "pp  SPY-EEM:", round(100 * dose_eem.iloc[-1], 2))

    for name, c in (("DX", dx), ("UUP", uup)):
        a_me = on_cal(c, me[me >= c.index[0] + pd.Timedelta(days=10)])
        me_map = pd.Series(me[me >= c.index[0] + pd.Timedelta(days=10)], index=a_me)
        rows = []
        for h in HS:
            f = fwd(c, a_me, h)
            f.index = me_map.reindex(f.index).values
            q = f[f.index.month.isin([3, 6, 9, 12])]
            o = f[~f.index.month.isin([3, 6, 9, 12])]
            allw = all_windows(c, h)
            rows += [rec(q, f"{name} h={h} POST-QE long"), rec(o, f"{name} h={h} ordinary ME"),
                     rec(allw, f"{name} h={h} all days")]
            d1, t1 = welch(q, o)
            d2, t2 = welch(q, allw)
            rows.append({"label": f"   QE-ordME {d1:+.3f}pp t{t1:+.2f} | QE-alldays {d2:+.3f}pp t{t2:+.2f}"})
        show(rows, f"{name} long from the QE close")

    # focus on DX
    a_me = on_cal(dx, me)
    me_map = pd.Series(me, index=a_me)
    for h in (3, 5, 10):
        f = fwd(dx, a_me, h); f.index = me_map.reindex(f.index).values
        q = f[f.index.month.isin([3, 6, 9, 12])]
        dq = asof_on(dose, q.index)
        dq2 = asof_on(dose_eem, q.index)
        print(f"\n--- DX h={h}: {len(q)} quarter-ends ---")
        print("  DOSE SPY-EFA -> post-QE DX:", regress(dq, q))
        print("  DOSE SPY-EEM -> post-QE DX:", regress(dq2, q))
        o = f[~f.index.month.isin([3, 6, 9, 12])]
        print("  DOSE SPY-EFA on ORDINARY ME:", regress(asof_on(dose, o.index), o))
        ok = dq.notna()
        t3 = pd.qcut(dq[ok], 3, labels=["US lagged", "mid", "US led"])
        show([rec(q[ok][t3 == l], f"h={h} dose tercile {l}") for l in ["US lagged", "mid", "US led"]]
             + [rec(q[ok][dq[ok] > 0], "dose > 0"), rec(q[ok][dq[ok] > 0.0195], "dose > +1.95pp (today)")],
             "dose terciles (pre-specified)")
        show(eras(q, f"h={h} QE", ("2008-01-01", "2018-01-01")), "era")
        mid = q.index.year % 4 == 2
        show([rec(q[mid], "midterm"), rec(q[~mid], "non-midterm")]
             + [rec(q[q.index.month == m], f"month {m}") for m in (3, 6, 9, 12)], "cycle / quarter")
        print("  concentration:", cluster_note(q.index, q.values))

    # mechanism leg: does the dose predict dollar weakness INTO the QE close?
    run = fwd(dx, a_me, 5, start=-5); run.index = me_map.reindex(run.index).values
    rq = run[run.index.month.isin([3, 6, 9, 12])]
    print("\nMECHANISM LEG: dose -> DX QE-5->QE (hedge selling predicts NEGATIVE slope):",
          regress(asof_on(dose, rq.index), rq))
    ro = run[~run.index.month.isin([3, 6, 9, 12])]
    print("   same on ordinary ME:", regress(asof_on(dose, ro.index), ro))
    run0 = fwd(dx, a_me, 1, start=-1); run0.index = me_map.reindex(run0.index).values
    r0q = run0[run0.index.month.isin([3, 6, 9, 12])]
    print("   dose -> DX QE-0 session:", regress(asof_on(dose, r0q.index), r0q))

    # state: DX at its 252 high / z10
    dmax = rolling_on_valid(dx, lambda x: x.rolling(252).max())
    z10 = zscore(dx, 10)
    for h in (3, 5, 10):
        f = fwd(dx, a_me, h); f.index = me_map.reindex(f.index).values
        q = f[f.index.month.isin([3, 6, 9, 12])]
        near = asof_on(dx >= 0.995 * dmax, pd.DatetimeIndex(a_me)).reindex(pd.DatetimeIndex(a_me))
        near = pd.Series(near.values, index=me_map.values).reindex(q.index).fillna(False).astype(bool)
        zz = pd.Series(asof_on(z10, pd.DatetimeIndex(a_me)).values, index=me_map.values).reindex(q.index)
        show([rec(q[near], f"h={h} QE with DX within 0.5% of 252 high"), rec(q[~near], "  not near high"),
              rec(q[zz >= 2], f"h={h} QE with DX z10 >= 2"), rec(q[zz >= 1], f"h={h} QE with DX z10 >= 1"),
              rec(q[near & (zz >= 1.5)], "near high AND z10>=1.5")],
             "today's DX state inside the post-QE cell")
        if h == 5:
            print("  near-high QE dates:", ", ".join(f"{d.date()}:{100*v:+.2f}" for d, v in q[near].items()))
