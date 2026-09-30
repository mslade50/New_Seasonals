"""c9 round 2 on the ONE positive row round 1 surfaced: the GIS-like cross
(63d winner r63 >= 80 that is a 21d laggard r21 <= 15) long into the
pre-print session (k=2: entry T-2, exit T-1, h=1). That row was one of 16 cross
rows scanned (4 crosses x 2 k x 2 universes), so it is a SEARCHED cell.

Questions: (1) is the print doing anything, or is this the generic
pullback-in-uptrend reversal (the book's LT Trend ST OS family)? -> the
NO-PRINT placebo with the same cross gate, paired by week; (2) offset ladder
on the cross; (3) definition neighbours; (4) era / regime; (5) GIS's own
record in the cross.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa
import importlib.util

spec = importlib.util.spec_from_file_location("c9", Path(__file__).parent / "kB_c9_r1.py")
# reuse builders without re-running the round-1 prints: copy the two functions
P = build_panel()
C, beta, r5, r21, r63 = (P[k] for k in ("C", "beta", "r5", "r21", "r63"))
idx = C.index
spy = C["SPY"].values
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]
V, B = C.values, beta.values
R5, R21, R63 = r5.values, r21.values, r63.values


def build(univ, kent, shifts=range(-10, 6)):
    Ev = event_positions(E, idx, univ)
    ci = np.array([C.columns.get_loc(t) for t in Ev.ticker])
    pT = Ev.pT.values
    out = []
    for j in shifts:
        ent = pT + j - kent
        sig = ent - 1
        ex = pT + j - 1
        ok = (sig >= 252) & (ex < len(idx)) & (ex > ent)
        s, e, x, c = sig[ok], ent[ok], ex[ok], ci[ok]
        rn = V[x, c] / V[e, c] - 1.0
        rs = spy[x] / spy[e] - 1.0
        out.append(pd.DataFrame({
            "j": j, "ticker": C.columns[c], "entry_date": idx[e], "T": idx[pT[ok]],
            "r5": R5[s, c], "r21": R21[s, c], "r63": R63[s, c],
            "long": rn, "bh": rn - B[s, c] * rs,
            "spy200": P["spy_above200"].values[s]}))
    D = pd.concat(out, ignore_index=True).dropna(subset=["long", "bh", "r63", "r21"])
    D["year"] = D.entry_date.dt.year
    return D


def placebo(univ, h, g):
    Cu = C[univ]
    fwd = Cu.shift(-(1 + h)) / Cu.shift(-1) - 1.0
    spf = C["SPY"].shift(-(1 + h)) / C["SPY"].shift(-1) - 1.0
    bh = fwd - beta[univ].mul(spf, axis=0)
    pm = print_mask(event_positions(E, idx, univ), idx, univ)
    near = pm.astype(float).rolling(h + 4, min_periods=1).sum().shift(-(h + 2)) > 0
    first = E[E.ticker.isin(univ)].groupby("ticker").date.min()
    last = E[E.ticker.isin(univ)].groupby("ticker").date.max()
    cov = pd.DataFrame({t: (idx >= first.get(t, idx[-1])) & (idx <= last.get(t, idx[0]))
                        for t in univ}, index=idx)
    m = g & ~near & cov & bh.notna()
    st = bh.where(m).stack()
    pos = idx.get_indexer(st.index.get_level_values(0))
    ent = idx[np.minimum(pos + 1, len(idx) - 1)]
    return week_cluster(ent, st.values), len(st)


for name, univ in [("LIQ", LIQ_U), ("BROAD", [t for t in C.columns if t != "SPY"])]:
    for kent in (2, 3, 4):
        h = kent - 1
        D = build(univ, kent)
        D0 = D[D.j == 0]
        X = D0[(D0.r63 >= 80) & (D0.r21 <= 15)]
        pw, npl = placebo(univ, h, (r63[univ] >= 80) & (r21[univ] <= 15))
        gw = week_cluster(X.entry_date, X.bh)
        j = pd.concat([gw.rename("p"), pw.rename("n")], axis=1, join="inner")
        rows = [cl_stat(X, "bh", f"CROSS print cell k={kent} h={h} [bh]"),
                cl_stat(X, "long", f"CROSS print cell [long]"),
                dict(stat(pw.values, "NO-PRINT placebo, same cross [bh]"), n_obs=npl),
                stat((j.p - j.n).values, "PAIRED same-week print minus no-print [bh]")]
        show(rows, f"[{name}] k={kent}: print vs no-print for the cross")
        if kent != 2:
            continue
        rows = []
        for lbl, m in [("pre-2018", X.year < 2018), ("2018+", X.year >= 2018),
                       ("midterm", X.year % 4 == 2), ("SPY>200d", X.spy200.astype(bool)),
                       ("SPY<200d", ~X.spy200.astype(bool)), ("Sep/Oct T", X["T"].dt.month.isin([9, 10]))]:
            rows.append(cl_stat(X[m], "bh", lbl))
        show(rows, f"[{name}] k=2 cross: era / regime")
        print("  concentration:", cluster_note(gw.index.to_timestamp(), gw.values))
        rows = []
        for a, b in [(80, 5), (80, 10), (80, 15), (80, 20), (80, 25), (70, 15), (90, 15),
                     (95, 15), (0, 15), (50, 15)]:
            rows.append(cl_stat(D0[(D0.r63 >= a) & (D0.r21 <= b)], "bh", f"r63>={a} & r21<={b}"))
        show(rows, f"[{name}] k=2 cross: definition neighbours (bh)")
        rows = []
        for jj in range(-10, 6):
            Dj = D[(D.j == jj) & (D.r63 >= 80) & (D.r21 <= 15)]
            rows.append(cl_stat(Dj, "bh", f"j={jj:+d}"))
        L = pd.DataFrame(rows)
        L["rank"] = L.mean_pct.rank(ascending=False).astype(int)
        show(L[["label", "n", "n_obs", "mean_pct", "t", "hit", "rec", "sign_p", "rank"]]
             .to_dict("records"), f"[{name}] k=2 cross: OFFSET LADDER (bh)")
        k0 = L[L.label == "j=+0"].iloc[0]
        print(f"  true rank {int(k0['rank'])} of {len(L)}; true {k0.mean_pct:+.3f}% vs ladder "
              f"ex-true mean {L[L.label != 'j=+0'].mean_pct.mean():+.3f}%")

D = build(["GIS"], 2, shifts=[0])
X = D[(D.r63 >= 80) & (D.r21 <= 15)]
print("\nGIS own record in the cross (k=2):")
print(X[["entry_date", "T", "r63", "r21", "long", "bh"]].round(4).to_string(index=False))
show([stat(X.bh.values, "GIS cross [bh]"), stat(X.long.values, "GIS cross [long]")])
