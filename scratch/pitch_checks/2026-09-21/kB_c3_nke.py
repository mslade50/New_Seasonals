"""c3 supplement: (1) NKE's own record at the live anchor (short T-8 -> T-1),
all prints and gated; (2) pooled LIQ gated cell, horizon inside the pre-print
window (entry T-8, exit T-8+h, h=1..7) -- does the laggard fall at ANY hold?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kB_common import *  # noqa

P = build_panel()
C, beta, lodist, r21 = P["C"], P["beta"], P["lodist"], P["r21"]
idx = C.index
spy = C["SPY"].values
E = P["E"]
LIQ_U = [t for t in LIQ if t in C.columns]


def rows(univ, h):
    Ev = event_positions(E, idx, univ)
    ci = np.array([C.columns.get_loc(t) for t in Ev.ticker])
    sig = Ev.pT.values - 9
    ent = sig + 1
    ex = ent + h
    ok = (sig >= 252) & (Ev.pT.values - 1 < len(idx)) & (ex <= Ev.pT.values - 1)
    s, e, x, c = sig[ok], ent[ok], ex[ok], ci[ok]
    V = C.values
    rn = V[x, c] / V[e, c] - 1
    rs = spy[x] / spy[e] - 1
    return pd.DataFrame({"ticker": C.columns[c], "entry_date": idx[e],
                         "lodist": lodist.values[s, c], "r21": r21.values[s, c],
                         "short": -rn, "bh": -(rn - beta.values[s, c] * rs)}).dropna()


N = rows(["NKE"], 7)
G = N[(N.lodist <= 0.03) & (N.r21 <= 15)]
show([stat(N.short.values, "NKE all prints short T-8->T-1 [raw]"),
      stat(N.bh.values, "NKE all prints [bh]"),
      stat(G.short.values, "NKE gated (lodist<=3%, r21<=15) [raw]"),
      stat(G.bh.values, "NKE gated [bh]")], "NKE own record")
print(G.round(4).to_string(index=False))

out = []
for h in range(1, 8):
    D = rows(LIQ_U, h)
    D = D[(D.lodist <= 0.03) & (D.r21 <= 15)]
    r = cl_stat(D, "bh", f"h={h} (exit T-{8 - h})")
    r["raw_mean_pct"] = round(cl_stat(D, "short", "")["mean_pct"], 3)
    out.append(r)
show(out, "LIQ gated: short from T-8, horizon inside the pre-print window (week-clustered)")
