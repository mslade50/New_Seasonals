"""Drill 09 found 9 of 26 'Wednesday after September quad witching' slots were
FOMC decision days, and drill 03's TLT top-two years (2011 Twist, 2022) are both
among them. Tomorrow is not an FOMC day (the decision was 2026-09-16). Re-run
TLT and the S&P on the slot, and TLT anchor -> September month-end, ex-FOMC."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

cp = close_panel(["TLT", "^GSPC", "^TNX"])
ev = load_events(["fomc_decision"])
fomc = set(pd.to_datetime(ev["date"]).dt.normalize())


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


for tkr in ("TLT", "^GSPC", "^TNX"):
    s = cp[tkr].dropna()
    idx = s.index
    recs = []
    for y in sorted(set(idx.year)):
        qf = third_friday(y, 9)
        pos = idx.searchsorted(qf)
        if pos + 3 >= len(idx) or idx[pos].year != y:
            continue
        a, w = idx[pos + 2], idx[pos + 3]  # anchor = 2nd session after expiry, w = next session
        if w >= idx[-1] + pd.Timedelta(days=1) or w > idx[-1]:
            continue
        mend = idx[(idx.year == y) & (idx.month == 9)][-1]
        win_fomc = any(d in fomc for d in idx[(idx > a) & (idx <= mend)])
        if tkr == "^TNX":
            h1, toend = 100 * (s[w] - s[a]), 100 * (s[mend] - s[a])  # bp
        else:
            h1, toend = s[w] / s[a] - 1, s[mend] / s[a] - 1
        recs.append(dict(a=a, w=w, fomc_w=w in fomc, fomc_win=win_fomc, h1=h1, toend=toend))
    df = pd.DataFrame(recs).set_index("a")
    print(f"\n######## {tkr}  (anchor = 2nd session after Sep quad Friday; h1 = next session)")
    if tkr == "^TNX":
        for lab, sub in (("all", df), ("ex-FOMC slot", df[~df.fomc_w]), ("ex-FOMC window", df[~df.fomc_win])):
            pre = sub.index < "2018-01-01"
            print(f"  {lab:15s} n={len(sub)} h1 {sub.h1.mean():+.2f}bp down {int((sub.h1<0).sum())} | "
                  f"toend {sub.toend.mean():+.2f}bp down {int((sub.toend<0).sum())} | "
                  f"pre-2018 toend {sub.toend[pre].mean():+.2f}bp down {int((sub.toend[pre]<0).sum())}/{pre.sum()} "
                  f"| 2018+ {sub.toend[~pre].mean():+.2f}bp down {int((sub.toend[~pre]<0).sum())}/{(~pre).sum()}")
        continue
    rows = []
    for lab, sub in (("all", df), ("ex-FOMC slot", df[~df.fomc_w]), ("FOMC slot only", df[df.fomc_w])):
        r = summarize(sub.h1.values, f"h1 {lab}")
        r["sign_p_up"] = sign_test(int((sub.h1 > 0).sum()), len(sub))
        r["sign_p_dn"] = sign_test(int((sub.h1 < 0).sum()), len(sub))
        rows.append(r)
    for lab, sub in (("all", df), ("ex-FOMC window", df[~df.fomc_win])):
        r = summarize(sub.toend.values, f"to Sep end {lab}")
        r["sign_p_up"] = sign_test(int((sub.toend > 0).sum()), len(sub))
        r["sign_p_dn"] = sign_test(int((sub.toend < 0).sum()), len(sub))
        rows.append(r)
    show(rows)
    ex = df[~df.fomc_w]
    show(era_split(ex.index, ex.h1.values), "h1 ex-FOMC era")
    print("  h1 ex-FOMC", cluster_note(ex.index, ex.h1.values))
    exw = df[~df.fomc_win]
    show(era_split(exw.index, exw.toend.values), "to Sep end ex-FOMC-window era")
    print("  per-year h1 %:", {d.year: (round(100 * v, 2), 'F' if f else '') for d, v, f in zip(df.index, df.h1, df.fomc_w)})
    # all-days control
    c1 = s.pct_change().shift(-1).dropna()
    print(f"  CTRL all days h1 mean {100*c1.mean():+.3f}% up {100*(c1>0).mean():.1f}%")
    wd = c1[c1.index.map(lambda d: (d + pd.Timedelta(days=1)).weekday() == 2 if True else False)]
