"""C2 round 2: (a) session decomposition of the post-ME TLT give-back, NFP vs
non-NFP sessions; (b) gate attribution and neighbours (kept vs deleted anchors,
fixed-calendar anchors so no re-anchoring is possible); (c) dose: does the
ME-5->ME run-in (the borrowed extension demand) predict the give-back;
(d) extension-heavy months (Feb/May/Aug/Nov refunding month-ends) vs rest;
(e) concentration, LOYO, era of the ex-NFP window."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kA_common import *  # noqa


def session_table(c: pd.Series, me: pd.DatetimeIndex, nfp: set, K: int = 10):
    idx = c.index
    pos = pd.Series(range(len(idx)), index=idx)
    rows = []
    for d in me:
        p = pos[d]
        for k in range(1, K + 1):
            if p + k >= len(idx):
                continue
            r = -(c.iloc[p + k] / c.iloc[p + k - 1] - 1.0)  # SHORT
            rows.append({"me": d, "k": k, "date": idx[p + k], "r": r,
                         "nfp": idx[p + k] in nfp})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tlt, tnx = own_series("TLT"), own_series("^TNX")
    nfp = set(nfp_dates())
    me = month_ends(tlt.index)
    base1 = -(tlt.pct_change().dropna())
    print(f"all-days one-session SHORT TLT: {100*base1.mean():+.4f}%  (n={len(base1)})")
    nfp_all = base1.reindex([d for d in nfp if d in base1.index]).dropna()
    print(f"ALL NFP sessions SHORT TLT: {100*nfp_all.mean():+.4f}%  n={len(nfp_all)} "
          f"t={nfp_all.mean()/(nfp_all.std()/np.sqrt(len(nfp_all))):+.2f}")

    S = session_table(tlt, me, nfp)
    rows = []
    for k in range(1, 11):
        s = S[S.k == k]
        a, b = s[s.nfp].r, s[~s.nfp].r
        r = rec(s.r, f"ME+{k} session")
        r["nfp_share"] = round(100 * s.nfp.mean(), 1)
        r["nfp_mean"] = round(100 * a.mean(), 3) if len(a) else np.nan
        r["non_nfp_mean"] = round(100 * b.mean(), 3)
        r["non_nfp_t"] = round(b.mean() / (b.std() / np.sqrt(len(b))), 2)
        rows.append(r)
    show(rows, "TLT short, one session at a time after ME (percent)")

    for lo, hi, lab in ((1, 5, "ME+1..+5"), (1, 7, "ME+1..+7"), (1, 10, "ME+1..+10")):
        s = S[(S.k >= lo) & (S.k <= hi)]
        a, b = s[s.nfp].r, s[~s.nfp].r
        show([rec(a, f"{lab} NFP sessions"), rec(b, f"{lab} NON-NFP sessions")],
             f"pooled sessions {lab} vs all-days {100*base1.mean():+.4f}%")
    # era of the non-NFP sessions
    s = S[(S.k <= 7)]
    for lo, hi in (("2002", "2013"), ("2013", "2020"), ("2020", "2027")):
        m = (s.date >= lo) & (s.date < hi)
        a, b = s[m & s.nfp].r, s[m & ~s.nfp].r
        print(f"  {lo}-{int(hi)-1}: NFP sess {100*a.mean():+.3f}% (n{len(a)})   non-NFP sess "
              f"{100*b.mean():+.3f}% (n{len(b)}, t {b.mean()/(b.std()/np.sqrt(len(b))):+.2f})  "
              f"all-days {100*base1[(base1.index>=lo)&(base1.index<hi)].mean():+.3f}%")

    # h=5 window: NFP inside vs not, and the ex-NFP window (sum of non-NFP sessions)
    for h in (3, 5, 7):
        w = -fwd(tlt, me, h)
        inside = pd.Series(event_in_window(w.index, tlt.index, h, 0, ("nfp",)), index=w.index)
        ex = S[(S.k <= h) & (~S.nfp)].groupby("me").r.sum().reindex(w.index)
        show([rec(w[inside], f"h={h} NFP inside window"), rec(w[~inside], f"h={h} NFP NOT inside"),
              rec(ex, f"h={h} window EX the NFP session (sum non-NFP)")]
             + eras(ex, f"h={h} ex-NFP", ("2013-01-01", "2018-01-01", "2020-01-01")),
             f"h={h} NFP split")

    # (b) gate attribution + neighbours at h=5 and h=10
    tmax = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
    rk63 = pct_rank(tnx, 63)
    tlt_lo = rolling_on_valid(tlt, lambda x: x.rolling(252).min())
    for h in (5, 10):
        w = -fwd(tlt, me, h)
        gates = {}
        for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
            gates[f"TNX within {100*thr:.1f}% of 252max"] = asof_on(tnx >= (1 - thr) * tmax, w.index).fillna(False).astype(bool)
        gates["TNX 63d-chg rank >= 90"] = asof_on(rk63 >= 90, w.index).fillna(False).astype(bool)
        gates["TLT within 1% of 252 low"] = (tlt.reindex(w.index) <= 1.01 * tlt_lo.reindex(w.index))
        rows = [rec(w, f"h={h} PARENT")]
        for lab, g in gates.items():
            g = g.reindex(w.index).fillna(False).astype(bool)
            r = rec(w[g], f"KEPT  {lab}")
            r["deleted_pct"] = round(100 * w[~g].mean(), 3)
            rows.append(r)
        show(rows, f"gate neighbours h={h} (kept vs deleted)")

    # (c) dose: run-in ME-5->ME long TLT vs post-ME short h=5
    for h in (5, 7):
        w = -fwd(tlt, me, h)
        run = fwd(tlt, me, 5, start=-5).reindex(w.index)
        ok = run.notna() & w.notna()
        b = np.polyfit(run[ok], w[ok], 1)
        rho = pd.Series(run[ok]).corr(w[ok], method="spearman")
        n = ok.sum()
        resid = w[ok] - np.polyval(b, run[ok])
        se = np.sqrt(resid.var(ddof=2) / ((run[ok] - run[ok].mean())**2).sum())
        print(f"\nDOSE h={h}: slope {b[0]:+.4f} (t {b[0]/se:+.2f}), spearman {rho:+.3f}, n {n}")
        q = pd.qcut(run[ok], 3, labels=["run-in low", "mid", "high"])
        show([rec(w[ok][q == l], f"h={h} {l} tercile") for l in ["run-in low", "mid", "high"]],
             "dose terciles")

    # (d) extension-heavy month-ends (Feb/May/Aug/Nov refunding) vs rest
    for h in (5, 7):
        w = -fwd(tlt, me, h)
        big = w.index.month.isin([2, 5, 8, 11])
        qe = w.index.month.isin([3, 6, 9, 12])
        show([rec(w[big], f"h={h} Feb/May/Aug/Nov (big extension)"),
              rec(w[qe], f"h={h} Mar/Jun/Sep/Dec (QE)"),
              rec(w[~big & ~qe], f"h={h} other"),
              ] + [rec(w[w.index.month == m], f"h={h} month {m}") for m in range(1, 13)],
             f"month split h={h}")

    # (e) concentration + LOYO at h=5
    w = -fwd(tlt, me, 5)
    print("\nconcentration h=5:", cluster_note(w.index, w.values))
    by = w.groupby(w.index.year).mean()
    loyo = [(y, 100 * w[w.index.year != y].mean()) for y in by.index]
    print("LOYO h=5 min/max: %+.3f / %+.3f" % (min(v for _, v in loyo), max(v for _, v in loyo)))
    print("years positive: %d of %d" % ((w.groupby(w.index.year).sum() > 0).sum(), len(by)))
    srt = w.sort_values(ascending=False)
    print("drop best 5 episodes h=5: %+.3f%%" % (100 * srt.iloc[5:].mean()))
