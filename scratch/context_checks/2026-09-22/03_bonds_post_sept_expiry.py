"""The engine's Sep-23 seasonal says TLT up 17 of 23 and the 10y yield down 19 of
26. Re-anchor it structurally: the 2nd session after the September quad-witching
Friday (today), then h1 (Wednesday) and anchor -> last September session.
Controls: the same anchor after the Mar/Jun/Dec expiries, and all days."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

px = load_prices(["TLT", "IEF", "^TNX", "SPY"])
tlt = px["TLT"]["Close"].dropna()
tnx = px["^TNX"]["Close"].dropna()
idx = tnx.index


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def anchors(month: int, series_idx: pd.DatetimeIndex) -> dict:
    out = {}
    for y in range(1999, 2027):
        qf = third_friday(y, month)
        pos = series_idx.searchsorted(qf)  # quad friday or next session if holiday
        if pos + 2 >= len(series_idx) or series_idx[pos].year != y:
            continue
        a = series_idx[pos + 2]
        # last session of the month
        in_m = series_idx[(series_idx.year == y) & (series_idx.month == month)]
        if len(in_m) == 0 or a > in_m[-1]:
            continue
        mend = in_m[-1]
        out[a] = mend
    return out


def tnx_bp(a, b):
    return 100 * (tnx.loc[b] - tnx.loc[a]) / 10 if tnx.loc[a] > 20 else 100 * (tnx.loc[b] - tnx.loc[a])


print("TNX today", tnx.iloc[-1], "scale check (yield pct):", tnx.tail(3).values)

for month in (9, 3, 6, 12):
    an = anchors(month, tlt.index)
    an = {a: e for a, e in an.items() if a < tlt.index[-1]}
    nxt = {a: tlt.index[tlt.index.get_loc(a) + 1] for a in an}
    h1 = np.array([tlt.loc[nxt[a]] / tlt.loc[a] - 1 for a in an])
    toend = np.array([tlt.loc[e] / tlt.loc[a] - 1 for a, e in an.items()])
    dts = pd.DatetimeIndex(list(an))
    rows = [summarize(h1, f"TLT h1 m{month}"), summarize(toend, f"TLT to month-end m{month}")]
    show(rows, f"expiry month {month}, anchor = 2nd session after quad Friday")
    if month == 9:
        show(era_split(dts, h1), "Sep TLT h1 era")
        show(era_split(dts, toend), "Sep TLT to-month-end era")
        print("  h1", cluster_note(dts, h1))
        print("  toend", cluster_note(dts, toend))
        mid = dts.year % 4 == 2
        show([summarize(h1[mid], "midterm h1"), summarize(toend[mid], "midterm toend")], "Sep midterm")
        print("  per-year toend %:", {d.year: round(100 * v, 2) for d, v in zip(dts, toend)})
        print("  sign p up h1:", sign_test(int((h1 > 0).sum()), len(h1)),
              " toend:", sign_test(int((toend > 0).sum()), len(toend)))
        # all 6-session windows control
        ctrl6 = fwd_ret(tlt, 6).dropna()
        print(f"  CTRL all 6-session TLT windows mean {100*ctrl6.mean():.3f}% hit {100*(ctrl6>0).mean():.1f}%")
        ctrl1 = fwd_ret(tlt, 1).dropna()
        print(f"  CTRL all days TLT h1 mean {100*ctrl1.mean():.3f}% hit {100*(ctrl1>0).mean():.1f}%")
        # 10y yield bp, 1999+
        an2 = anchors(9, idx)
        an2 = {a: e for a, e in an2.items() if a < idx[-1]}
        bp_end = np.array([100 * (tnx.loc[e] - tnx.loc[a]) for a, e in an2.items()])
        nx2 = [idx[idx.get_loc(a) + 1] for a in an2]
        bp_h1 = np.array([100 * (tnx.loc[n] - tnx.loc[a]) for a, n in zip(an2, nx2)])
        d2 = pd.DatetimeIndex(list(an2))
        print(f"  10y bp h1: n={len(bp_h1)} mean {bp_h1.mean():+.2f}bp down {int((bp_h1<0).sum())}"
              f" | to month-end mean {bp_end.mean():+.2f}bp median {np.median(bp_end):+.2f} down {int((bp_end<0).sum())}")
        pre = d2 < pd.Timestamp("2018-01-01")
        print(f"    pre-2018 toend mean {bp_end[pre].mean():+.2f}bp down {int((bp_end[pre]<0).sum())}/{pre.sum()}"
              f" | 2018+ mean {bp_end[~pre].mean():+.2f}bp down {int((bp_end[~pre]<0).sum())}/{(~pre).sum()}")
        # condition: 10y up over prior 63 sessions by >= 40bp (today ~ +48bp?)
        chg63 = tnx.diff(63)
        print(f"  today 10y 63d change: {100*chg63.iloc[-1]:+.1f}bp")
        hot = np.array([chg63.loc[a] >= 0.30 for a in an2])
        print(f"    anchors with 10y up >=30bp over 63d: n={hot.sum()} toend mean {bp_end[hot].mean():+.2f}bp"
              f" down {int((bp_end[hot]<0).sum())} ; others mean {bp_end[~hot].mean():+.2f}bp")
        print("    hot years:", [d.year for d, h in zip(d2, hot) if h],
              [round(v, 1) for v, h in zip(bp_end, hot) if h])
