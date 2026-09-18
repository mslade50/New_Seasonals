"""C7 round 1b: the calm-shock parent is near zero at the live dose. Measure
(1) the dose ladder in ATR buckets with era stability (the <= -2 ATR bucket
    looked like drift; today's names sit at -1.69 / -1.70 / -1.83),
(2) whether C2's state (r63<=5 & 252d>=+40% & r5<15) and C4's state (CL=F
    r5>=90) earn a REVERSAL against the parent,
(3) cost per unit member notional at the live 126d betas (2.5 bp/leg per
    unit notional traded, hedge notional = |beta|).
Short residual = -(member fwd - beta*SPY fwd), lag=1, per-name gap 5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, REF23, date_clusters, rec, vret  # noqa

R = pd.read_pickle(Path(__file__).with_name("k1_c7_shocks.pkl"))
px = load_prices(REF23 + ["SPY", "CL=F"])
nyse = px["SPY"].index[px["SPY"].index <= ASOF]
calm = []
for t, g in R[R.kind == "calm"].groupby("t"):
    dd = declusters(pd.DatetimeIndex(g.d), 5, nyse)
    calm.append(g[g.d.isin(dd)])
C = pd.concat(calm).reset_index(drop=True)
for h in (1, 3, 5):
    C[f"short{h}"] = -C[f"res{h}"]


def line(sub, col, label):
    x = summarize(sub[col].values, label)
    cl, _ = date_clusters(sub.d, sub[col].values, 7)
    x["clusters"], x["cl_pct"], x["cl_rec"] = len(cl), 100 * np.nanmean(cl) if len(cl) else np.nan, rec(cl)
    return x


C["bucket"] = pd.cut(C.move_atr, [-99, -2.5, -2.0, -1.75, -1.5],
                     labels=["<=-2.5", "-2.5..-2", "-2..-1.75", "-1.75..-1.5"])
rows = []
for b in ["-1.75..-1.5", "-2..-1.75", "-2.5..-2", "<=-2.5"]:
    S = C[C.bucket == b]
    for h in (3, 5):
        rows.append(line(S, f"short{h}", f"{b} h={h}"))
        rows.append(line(S[S.d < "2018-01-01"], f"short{h}", f"   {b} h={h} pre-2018"))
        rows.append(line(S[S.d >= "2018-01-01"], f"short{h}", f"   {b} h={h} 2018+"))
show(rows, "(1) DOSE LADDER, SHORT residual (positive = drift claim pays)")
from scipy.stats import spearmanr  # noqa
for h in (3, 5):
    v = C.dropna(subset=[f"short{h}"])
    print(f"  Spearman(move_atr, short{h}) = {spearmanr(v.move_atr, v[f'short{h}']).correlation:+.3f} on {len(v)}")
S2 = C[C.move_atr <= -2.0]
show([line(S2, "short5", "<=-2 ATR h=5 all"), line(S2[S2.spy200 >= 0], "short5", "  SPY>=200d"),
      line(S2[S2.spy200 < 0], "short5", "  SPY<200d"),
      line(S2[S2.d.dt.year != 2008], "short5", "  drop 2008"),
      line(S2[~S2.t.isin(["GDX", "FXI", "EEM"])], "short5", "  ex GDX/FXI/EEM")], "<= -2 ATR bucket detail")
print("  " + cluster_note(pd.DatetimeIndex(S2.dropna(subset=['short5']).d),
                          S2.dropna(subset=['short5']).short5.values, k=2))
live = C[(C.move_atr > -2.0)]
show([line(live, "short1", "LIVE DOSE (-2,-1.5] h=1"), line(live, "short3", "LIVE DOSE h=3"),
      line(live, "short5", "LIVE DOSE h=5"),
      line(live[live.spy200 >= 0], "short3", "LIVE DOSE h=3 SPY>=200d"),
      line(live[live.spy200 >= 0], "short5", "LIVE DOSE h=5 SPY>=200d"),
      line(live[live.dial >= 50], "short5", "LIVE DOSE h=5 dial>=50")], "today's dose bucket")

# (2) conditioners inside the parent
feat = {}
for t in REF23:
    s = px[t]["Close"]
    s = s[s.index <= ASOF]
    feat[t] = pd.DataFrame({"r63": pct_rank(s, 63), "r5": pct_rank(s, 5), "r252": vret(s, 252)})
cl = px["CL=F"]["Close"].dropna()
cl_r5 = pct_rank(cl, 5)
C["c2"] = [bool((feat[t].loc[d, "r63"] <= 5) and (feat[t].loc[d, "r252"] >= 0.40) and (feat[t].loc[d, "r5"] < 15))
           if d in feat[t].index else False for t, d in zip(C.t, C.d)]
C["cl_up"] = [float(cl_r5.asof(d)) >= 90 for d in C.d]
show([line(C[C.c2], "short5", "C2 state inside shock parent h=5"),
      line(C[C.c2], "short3", "C2 state inside shock parent h=3"),
      line(C[~C.c2], "short5", "not C2 h=5"),
      line(C[(C.t == "OIH") & C.cl_up], "short5", "OIH shock & CL r5>=90 h=5"),
      line(C[(C.t == "OIH") & ~C.cl_up], "short5", "OIH shock & CL r5<90 h=5"),
      line(C[C.t.isin(["OIH", "XLE", "XOP"]) & C.cl_up], "short5", "energy ETF shock & CL r5>=90 h=5"),
      line(C[C.t.isin(["OIH", "XLE", "XOP"]) & ~C.cl_up], "short5", "energy ETF shock & CL r5<90 h=5")],
     "(2) conditioners vs the parent (SHORT residual; negative = reversal earned)")
print(C[C.c2][["t", "d", "move_atr", "short3", "short5"]].round(4).to_string())

# (3) cost at live betas
for t, beta in (("SMH", 2.530), ("OIH", 0.340), ("EEM", 1.770)):
    rt = 2.5 * (1 + abs(beta))
    print(f"  {t}: beta {beta:.2f} -> round trip ~{rt:.1f} bp per unit member notional; 5x bar {5*rt:.0f} bp")
