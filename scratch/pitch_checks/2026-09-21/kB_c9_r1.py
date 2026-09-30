"""c9 round 1 (+ mandatory offset ladder up front): LONG a liquid 63d winner
(r63 >= 80 at the signal close) from T-k to the pre-print close T-1, pooled.

Live mapping (signal 09-18, entry MOC 09-21):
  GIS / PAYX  T = 09-23 -> entry T-2, exit T-1 -> h=1   (k=2)
  CAG         T = 09-30 -> entry T-7, exit T-1 -> h=6   (k=7)
T = first session on/after the calendar date; T-1 exit never holds a print.
Rows: raw long, SPY-relative, beta-hedged (252d beta at the signal close);
all summaries WEEK-CLUSTERED. Controls: every print (ungated announcement
premium), own drift (long every name every day, same h), the NO-PRINT placebo
(same r63 gate, no print within the window), the paired same-week difference,
and the offset ladder j=-10..+5 (anchor shifted, same gate re-evaluated).
Cross rows: r63>=80 & r21<=15 (GIS: r21 11.1), r63>=80 & r21<=35 (PAYX 31.3).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

P = build_panel()
C, beta, lodist, r5, r21, r63 = (P[k] for k in ("C", "beta", "lodist", "r5", "r21", "r63"))
idx = C.index
spy = C["SPY"].values
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]
V, B = C.values, beta.values
R5, R21, R63 = r5.values, r21.values, r63.values
qe_sess = np.zeros(len(idx), bool)
per = idx.to_period("M")
lom = np.r_[per[1:] != per[:-1], False]
qe_sess[lom & np.isin(idx.month, [3, 6, 9, 12])] = True
qe_cum = np.cumsum(qe_sess)


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
        b = B[s, c]
        out.append(pd.DataFrame({
            "j": j, "ticker": C.columns[c], "entry_date": idx[e],
            "T": idx[pT[ok]], "r5": R5[s, c], "r21": R21[s, c], "r63": R63[s, c],
            "long": rn, "rel": rn - rs, "bh": rn - b * rs,
            "spy200": P["spy_above200"].values[s],
            "qe_in": (qe_cum[x] - qe_cum[e]) > 0}))
    D = pd.concat(out, ignore_index=True).dropna(subset=["long", "bh", "r63"])
    D["year"] = D.entry_date.dt.year
    return D


def noprint_placebo(univ, h, gate_fn):
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
    g = gate_fn(univ)
    res = {}
    for col, val in (("long", fwd), ("bh", bh)):
        m = g & ~near & cov & val.notna()
        st = val.where(m).stack()
        d = st.index.get_level_values(0)
        pos = idx.get_indexer(d)
        ent = idx[np.minimum(pos + 1, len(idx) - 1)]
        res[col] = week_cluster(ent, st.values)
    own = bh.where(cov).stack()
    res["own_bh"] = own.groupby(own.index.get_level_values(0).to_period("W-FRI")).mean()
    ownl = fwd.where(cov).stack()
    res["own_long"] = ownl.groupby(ownl.index.get_level_values(0).to_period("W-FRI")).mean()
    return res


def g63(univ):
    return r63[univ] >= 80


for name, univ in [("LIQ", LIQ_U), ("BROAD", [t for t in C.columns if t != "SPY"])]:
    for kent in (2, 7):
        h = kent - 1
        D = build(univ, kent)
        D0 = D[D.j == 0]
        G0 = D0[D0.r63 >= 80]
        print(f"\n{'=' * 78}\n[{name}] k={kent} (entry T-{kent}, exit T-1, h={h}): "
              f"prints {len(D0)}, gated {len(G0)}\n{'=' * 78}")
        rows = [cl_stat(G0, c, f"GATED r63>=80 long [{c}]") for c in ("long", "rel", "bh")]
        rows += [cl_stat(D0, c, f"ALL prints long [{c}]") for c in ("long", "bh")]
        rows += [cl_stat(D0[D0.r63 < 80], "bh", "complement r63<80 [bh]")]
        pl = noprint_placebo(univ, h, g63)
        rows.append(stat(pl["own_long"].values, f"OWN drift long all days h={h} [long]"))
        rows.append(stat(pl["own_bh"].values, f"OWN drift [bh]"))
        rows.append(stat(pl["long"].values, "NO-PRINT placebo r63>=80 [long]"))
        rows.append(stat(pl["bh"].values, "NO-PRINT placebo r63>=80 [bh]"))
        show(rows, f"[{name}] k={kent} 1. pattern vs baselines (week-clustered)")
        for col in ("long", "bh"):
            gw = week_cluster(G0.entry_date, G0[col])
            j = pd.concat([gw.rename("p"), pl[col].rename("n")], axis=1, join="inner")
            show([stat((j.p - j.n).values, f"PAIRED same-week print minus no-print [{col}]")])

        rows = []
        for lbl, m in [("pre-2018", G0.year < 2018), ("2018+", G0.year >= 2018),
                       ("midterm", G0.year % 4 == 2), ("non-midterm", G0.year % 4 != 2),
                       ("SPY>200d", G0.spy200.astype(bool)), ("SPY<200d", ~G0.spy200.astype(bool)),
                       ("Sep prints", G0["T"].dt.month == 9),
                       ("Sep/Oct prints", G0["T"].dt.month.isin([9, 10])),
                       ("QE in window", G0.qe_in.astype(bool)),
                       ("cross r21<=15 (GIS-like)", G0.r21 <= 15),
                       ("cross r21<=35 (PAYX-like)", G0.r21 <= 35),
                       ("cross r21>50", G0.r21 > 50),
                       ("cross r5>=80 (CAG-like)", G0.r5 >= 80)]:
            rows.append(cl_stat(G0[m], "bh", f"{lbl} [bh]"))
        show(rows, f"[{name}] k={kent} 2. era / regime / cross rows (gated, bh)")
        if len(G0):
            gw = week_cluster(G0.entry_date, G0.bh)
            print("  concentration:", cluster_note(gw.index.to_timestamp(), gw.values))
        rows = []
        for thr in (50, 70, 80, 90, 95):
            rows.append(cl_stat(D0[D0.r63 >= thr], "bh", f"r63>={thr} [bh]"))
        for lo_, hi_ in ((0, 20), (20, 50), (50, 80), (80, 101)):
            rows.append(cl_stat(D0[(D0.r63 >= lo_) & (D0.r63 < hi_)], "bh", f"r63 in [{lo_},{hi_}) [bh]"))
        show(rows, f"[{name}] k={kent} 2b. gate gradient (bh)")

        rows = []
        for jj in range(-10, 6):
            Gk = D[(D.j == jj) & (D.r63 >= 80)]
            r = cl_stat(Gk, "bh", f"j={jj:+d}")
            r["long_mean_pct"] = cl_stat(Gk, "long", "").get("mean_pct")
            rows.append(r)
        L = pd.DataFrame(rows)
        L["rank_bh"] = L.mean_pct.rank(ascending=False).astype(int)
        L["rank_long"] = L.long_mean_pct.rank(ascending=False).astype(int)
        show(L[["label", "n", "n_obs", "mean_pct", "t", "hit", "rec", "rank_bh",
                "long_mean_pct", "rank_long"]].to_dict("records"),
             f"[{name}] k={kent} 3. OFFSET LADDER (gated; j=0 true; j>=1 exit on/after the print)")
        k0 = L[L.label == "j=+0"].iloc[0]
        oth = L[L.label != "j=+0"]
        pre = L[L.label.isin([f"j={x:+d}" for x in range(-10, 0)])]
        print(f"  true rank (bh) {int(k0.rank_bh)} of {len(L)}; true {k0.mean_pct:+.3f}% vs "
              f"ladder mean {oth.mean_pct.mean():+.3f}% (pre-print rungs only "
              f"{pre.mean_pct.mean():+.3f}%, max {pre.mean_pct.max():+.3f}%)")

# live names' own history at their live k
for t, kent in (("GIS", 2), ("PAYX", 2), ("CAG", 7)):
    D = build([t], kent, shifts=[0])
    rows = [stat(D.bh.values, f"{t} all own prints k={kent} [bh]"),
            stat(D[D.r63 >= 80].bh.values, f"{t} gated r63>=80 [bh]"),
            stat(D[D.r63 >= 80].long.values, f"{t} gated r63>=80 [long]")]
    show(rows, f"{t} own history")
    print(f"  live {t}: r5 {r5[t].iloc[-1]:.1f} r21 {r21[t].iloc[-1]:.1f} r63 {r63[t].iloc[-1]:.1f} "
          f"beta {beta[t].iloc[-1]:+.2f}")
