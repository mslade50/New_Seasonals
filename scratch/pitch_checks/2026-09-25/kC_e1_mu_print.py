"""E1 round 1: long a 63d-floor name that has already turned (r63 <= 5 AND r5 >= 70
at the signal close) from k=-3 into its own print, exiting at the pre-print close.

Reference class = LIQ single stocks (strategy_config.LIQUID_PLUS_COMMODITIES minus
ETFs, 162 names) on the earnings calendar (1996+). No usable BMO/AMC column
(timeOfTheDay is 99% null), so the class uses X = pT-1 (the close BEFORE the
announcement date, never holds through a print either way); entry X-3, h=3, state
measured at X-4. MU reports after the close, so MU's own rows use X = pT (the live
trade: signal 09-24, entry 09-25, exit 09-30).
Hedges: SPY-beta (252d) for the class; SMH-beta for semis. Week-clustered stats
(many names share a calendar week). Controls: all prints at the same offset (no
state gate), the SAME state with NO print within [entry, X+2] (placebo), and gate
attribution (each leg alone).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_e1_panel import *  # noqa
from kC_e1_panel import kc, get_panel, SEMIS
import numpy as np
import pandas as pd

P = get_panel()
C, beta, r5, r21, r63, E = (P[k] for k in ("C", "beta", "r5", "r21", "r63", "E"))
bsm = P["beta_smh"]
idx = C.index
LIQ = [t for t in kc.LIQ if t in C.columns]
V = C.values
spy = C["SPY"].values
smh = C["SMH"].values
col = {t: i for i, t in enumerate(C.columns)}


def build(univ, xoff=-1, k=3, shifts=(0,)):
    Ev = kc.event_positions(E, idx, univ)
    ci = np.array([col[t] for t in Ev.ticker])
    pT = Ev.pT.values
    out = []
    for j in shifts:
        X = pT + xoff + j
        ent = X - k
        sig = ent - 1
        ok = (sig >= 252) & (X < len(idx)) & (X > ent)
        s, e, x, c = sig[ok], ent[ok], X[ok], ci[ok]
        rn = V[x, c] / V[e, c] - 1.0
        rs = spy[x] / spy[e] - 1.0
        rsm = smh[x] / smh[e] - 1.0
        tk = C.columns[c]
        bs = np.array([bsm[t].values[si] if t in bsm.columns else np.nan for t, si in zip(tk, s)])
        out.append(pd.DataFrame({
            "j": j, "ticker": tk, "entry_date": idx[e], "T": idx[pT[ok]], "exit_date": idx[x],
            "r5": r5.values[s, c], "r21": r21.values[s, c], "r63": r63.values[s, c],
            "long": rn, "bh": rn - beta.values[s, c] * rs, "bh_smh": rn - bs * rsm,
            "spy200": P["spy_above200"].values[s]}))
    D = pd.concat(out, ignore_index=True).dropna(subset=["long", "bh", "r63", "r5"])
    D["year"] = D.entry_date.dt.year
    return D


def placebo(univ, h, g, col_="bh"):
    Cu = C[univ]
    fwd = Cu.shift(-(1 + h)) / Cu.shift(-1) - 1.0
    spf = C["SPY"].shift(-(1 + h)) / C["SPY"].shift(-1) - 1.0
    val = fwd - beta[univ].mul(spf, axis=0) if col_ == "bh" else fwd
    pm = kc.print_mask(kc.event_positions(E, idx, univ), idx, univ)
    near = pm.astype(float).rolling(h + 4, min_periods=1).sum().shift(-(h + 2)) > 0
    first = E[E.ticker.isin(univ)].groupby("ticker").date.min()
    last = E[E.ticker.isin(univ)].groupby("ticker").date.max()
    cov = pd.DataFrame({t: (idx >= first.get(t, idx[-1])) & (idx <= last.get(t, idx[0])) for t in univ},
                       index=idx)
    m = g & ~near & cov & val.notna()
    st = val.where(m).stack()
    pos = idx.get_indexer(st.index.get_level_values(0))
    ent = idx[np.minimum(pos + 1, len(idx) - 1)]
    return kc.week_cluster(ent, st.values), len(st)


GATE = lambda D: (D.r63 <= 5) & (D.r5 >= 70)  # noqa: E731
H = 3
D = build(LIQ, xoff=-1, k=H, shifts=range(-10, 6))
D0 = D[D.j == 0]
X = D0[GATE(D0)]
g_state = (r63[LIQ] <= 5) & (r5[LIQ] >= 70)
pw, npl = placebo(LIQ, H, g_state)
pw_long, _ = placebo(LIQ, H, g_state, "long")
gw = kc.week_cluster(X.entry_date, X.bh)
jn = pd.concat([gw.rename("p"), pw.rename("n")], axis=1, join="inner")
rows = [kc.cl_stat(X, "bh", f"E1 state print cell [bh SPY]"),
        kc.cl_stat(X, "long", "E1 state print cell [long]"),
        kc.cl_stat(D0, "bh", "ALL prints same offset, no gate [bh]"),
        kc.cl_stat(D0, "long", "ALL prints same offset, no gate [long]"),
        dict(kc.stat(pw.values, "NO-PRINT placebo, same state [bh]"), n_obs=npl),
        kc.stat(pw_long.values, "NO-PRINT placebo, same state [long]"),
        kc.stat((jn.p - jn.n).values, "PAIRED same-week print minus no-print [bh]")]
show(rows, f"1. LIQ class, entry X-3 -> X = pT-1 (h={H}), week-clustered")
print("  concentration (weeks):", cluster_note(gw.index.to_timestamp(), gw.values))
sv = np.sort(gw.values)
print(f"  drop-best-2 weeks mean {100*sv[:-2].mean():+.3f}%   obs-level mean {100*X.bh.mean():+.3f}% on {len(X)}")

show([kc.cl_stat(X[X.year < 2018], "bh", "pre-2018"), kc.cl_stat(X[X.year >= 2018], "bh", "2018+"),
      kc.cl_stat(X[X.spy200.astype(bool)], "bh", "SPY>200d"), kc.cl_stat(X[~X.spy200.astype(bool)], "bh", "SPY<200d"),
      kc.cl_stat(X[X.ticker.isin(SEMIS)], "bh", "semis in class [bh SPY]"),
      kc.cl_stat(X[X.ticker.isin(SEMIS)], "bh_smh", "semis in class [bh SMH]")], "2. era / regime / semis")

rows = []
for lbl, m in [("r63<=5 only", D0.r63 <= 5), ("r5>=70 only", D0.r5 >= 70), ("both (E1)", GATE(D0)),
               ("r63<=5 & r5<70", (D0.r63 <= 5) & (D0.r5 < 70)),
               ("r63<=3 & r5>=70", (D0.r63 <= 3) & (D0.r5 >= 70)), ("r63<=10 & r5>=70", (D0.r63 <= 10) & (D0.r5 >= 70)),
               ("r63<=5 & r5>=60", (D0.r63 <= 5) & (D0.r5 >= 60)), ("r63<=5 & r5>=80", (D0.r63 <= 5) & (D0.r5 >= 80)),
               ("r63<=10 & r5>=60", (D0.r63 <= 10) & (D0.r5 >= 60)), ("r63<=20 & r5>=70", (D0.r63 <= 20) & (D0.r5 >= 70))]:
    rows.append(kc.cl_stat(D0[m], "bh", lbl))
show(rows, "3. gate attribution + definition neighbours (bh SPY, week-clustered)")

rows = []
for jj in range(-10, 6):
    Dj = D[(D.j == jj)]
    rows.append(kc.cl_stat(Dj[GATE(Dj)], "bh", f"j={jj:+d}"))
L = pd.DataFrame(rows)
L["rank"] = L.mean_pct.rank(ascending=False).astype(int)
show(L[["label", "n", "n_obs", "mean_pct", "t", "hit", "rec", "sign_p", "rank"]].to_dict("records"),
     "4. offset ladder (window shifted j sessions; j=0 exits at pT-1)")

# other k (entry X-k), same exit
rows = []
for kk in (1, 2, 3, 4, 5):
    Dk = build(LIQ, xoff=-1, k=kk)
    rows.append(kc.cl_stat(Dk[GATE(Dk)], "bh", f"k={kk} (h={kk}) state"))
    rows.append(kc.cl_stat(Dk, "bh", f"k={kk} all prints"))
show(rows, "5. entry offset k (exit pT-1)")

# the live form: exit at pT (holds through BMO prints in the class; MU is AMC)
Dp = build(LIQ, xoff=0, k=3)
show([kc.cl_stat(Dp[GATE(Dp)], "bh", "exit pT (contaminated by BMO) state"),
      kc.cl_stat(Dp, "bh", "exit pT all prints")], "6. exit at pT close (live MU convention) for contrast")

# MU own prints, AMC: entry pT-3 -> pT
M = build(["MU"], xoff=0, k=3)
M["in_state"] = GATE(M)
print("\n7. MU own prints (entry pT-3 -> exit pT close, AMC), in-state rows:")
print(M[M.in_state][["entry_date", "T", "r5", "r63", "long", "bh", "bh_smh"]].round(4).to_string(index=False))
show([kc.stat(M.long.values, "MU all prints [long]"), kc.stat(M.bh_smh.values, "MU all prints [bh SMH]"),
      kc.stat(M[M.in_state].long.values, "MU in-state [long]"), kc.stat(M[M.in_state].bh_smh.values, "MU in-state [bh SMH]"),
      kc.stat(M[M.r63 <= 20].bh_smh.values, "MU r63<=20 [bh SMH]"), kc.stat(M[M.r5 >= 70].bh_smh.values, "MU r5>=70 [bh SMH]")])
mud = (C["MU"].shift(-4) / C["MU"].shift(-1) - 1).dropna()
print(f"  MU own 3-session drift (lag 1): {100*mud.mean():+.3f}%  hit {100*(mud > 0).mean():.1f}%")
