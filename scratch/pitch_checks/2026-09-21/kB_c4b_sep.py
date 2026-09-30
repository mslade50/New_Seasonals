"""c4 round 2: the one positive row round 1 surfaced is LIQ SEPTEMBER
(+1.036% bh, 15-4), a month picked from four quarter-ends. Test whether it is a
QUARTER-END object or just late-September weakness in weak names:
  (a) offset ladder: slide the whole 7-session window (and its QE-8 signal) by
      d = -10..+10 sessions around the September QE; rank d=0
  (b) the same ladder on ALL quarter-ends and on ordinary month-ends
  (c) BROAD minus LIQ (the non-liquid names) September as an independent sample
  (d) October month-end (mutual-fund fiscal year end Oct 31) as the adjacent
      tax-loss window
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

H = 7
P = build_panel()
C, beta, lodist = P["C"], P["beta"], P["lodist"]
idx = C.index
spy = C["SPY"].values
LIQ_U = [t for t in LIQ if t in C.columns]
per = idx.to_period("M")
me_pos = np.flatnonzero(np.r_[per[1:] != per[:-1], False])


def rows_at(univ, months, d, thr=0.02):
    ci = np.array([C.columns.get_loc(t) for t in univ])
    V, LD, B = C.values[:, ci], lodist.values[:, ci], beta.values[:, ci]
    out = []
    for m in me_pos:
        if idx[m].month not in months:
            continue
        x = m + d
        e = x - H
        s = e - 1
        if s < 252 or x >= len(idx):
            continue
        g = (LD[s] <= thr) & ~np.isnan(V[e]) & ~np.isnan(V[x])
        if not g.any():
            continue
        rn = V[x, g] / V[e, g] - 1.0
        rs = spy[x] / spy[e] - 1.0
        out.append({"me": idx[m], "year": idx[m].year, "n": int(g.sum()),
                    "bh": -np.nanmean(rn - B[s, g] * rs)})
    return pd.DataFrame(out)


broad = [t for t in C.columns if t != "SPY"]
nonliq = [t for t in broad if t not in set(LIQ_U)]
for name, univ in [("LIQ", LIQ_U), ("NONLIQ (BROAD minus LIQ)", nonliq)]:
    for lbl, months in [("SEPTEMBER", [9]), ("ALL QE", [3, 6, 9, 12]),
                        ("ORDINARY ME", [1, 2, 4, 5, 7, 8, 10, 11]), ("DECEMBER", [12])]:
        rows = []
        for d in range(-10, 11):
            M = rows_at(univ, months, d)
            r = stat(M.bh.values, f"d={d:+d}")
            if d >= 0 and months == [9]:
                r["label"] += " (window crosses into Oct)" if d > 0 else " (TRUE: ends on QE)"
            rows.append(r)
        L = pd.DataFrame(rows)
        L["rank"] = L.mean_pct.rank(ascending=False).astype(int)
        show(L[["label", "n", "mean_pct", "t", "hit", "rec", "sign_p", "rank"]].to_dict("records"),
             f"[{name}] {lbl}: window slid by d sessions around the month-end (bh short)")
        tru = L.iloc[10]
        print(f"  true (d=0) rank {int(tru['rank'])} of 21; true {tru.mean_pct:+.3f}% vs "
              f"ex-true mean {L.drop(10).mean_pct.mean():+.3f}%, d<0 mean "
              f"{L.iloc[:10].mean_pct.mean():+.3f}%")

    O = rows_at(univ, [10], 0)
    show([stat(O.bh.values, "OCTOBER ME-7 -> ME (fund fiscal YE) [bh]"),
          stat(O[O.year >= 2018].bh.values, "OCTOBER 2018+ [bh]")], f"[{name}] (d) October")
    S = rows_at(univ, [9], 0)
    print(f"  [{name}] Sep by year:", ", ".join(f"{y}:{100 * v:+.2f}(n{n})" for y, v, n in
                                             zip(S.year, S.bh, S.n)))
