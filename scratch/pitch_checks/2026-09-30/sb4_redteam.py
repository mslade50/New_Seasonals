"""sb4 red-team: TRV Oct 21d window (T+2 MOC entry). Recency, mechanism, regime, sizing, basket."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb1_engine as E

import numpy as np
import pandas as pd

T = "TRV"
PEERS = ["CB", "WRB", "PGR", "ALL", "HIG", "AIG"]
P = load_prices([T, "XLF", "SPY", "KRE", "FORM"] + PEERS)
f = P[T]
idx = f.index
H, L, C = (f[k].values for k in ("High", "Low", "Close"))
atr = wilder_atr(f["High"], f["Low"], f["Close"])
cl = pd.DataFrame({t: P[t]["Close"] for t in P}).reindex(idx)
anc = E.anchors(idx)

rows = []
for y, p in anc.items():
    e0, ex = p + 2, p + 23
    a = atr[p - 1]
    r = C[ex] / C[e0] - 1
    mae = (L[e0 + 1:ex + 1].min() - C[e0]) / a
    sep = C[p - 1] / C[p - 22] - 1
    spy = cl.SPY.iloc[:p]
    rw = {"year": y, "ret": r, "ret_atr": (C[ex] - C[e0]) / a, "mae_atr": mae, "sep": sep,
          "spy_sep": cl.SPY.iloc[p - 1] / cl.SPY.iloc[p - 22] - 1,
          "spy": cl.SPY.iloc[ex] / cl.SPY.iloc[e0] - 1, "xlf": cl.XLF.iloc[ex] / cl.XLF.iloc[e0] - 1,
          "spy_dd": cl.SPY.iloc[p - 1] / spy.iloc[-252:].max() - 1, "anchor": idx[p - 1]}
    for q in PEERS + ["KRE"]:
        v0, v1 = cl[q].iloc[e0], cl[q].iloc[ex]
        rw[q] = v1 / v0 - 1 if np.isfinite(v0) else np.nan
    # 252d betas: TRV on XLF alone, and on XLF + KRE
    rr = cl.pct_change().iloc[max(1, e0 - 254):e0 - 1]
    rw["bx"] = np.polyfit(rr.XLF, rr[T], 1)[0]
    if rr.KRE.notna().all() and len(rr) > 100:
        X = np.column_stack([np.ones(len(rr)), rr.XLF, rr.KRE])
        b = np.linalg.lstsq(X, rr[T].values, rcond=None)[0]
        rw["res_xk"] = r - b[1] * rw["xlf"] - b[2] * rw["KRE"]
    else:
        rw["res_xk"] = np.nan
    rows.append(rw)
W = pd.DataFrame(rows).set_index("year")
W["res_x"] = W.ret - W.bx * W.xlf
W["trv_spy"] = W.ret - W.spy
pc = lambda s: f"{100 * s.mean():+.2f}%"
rec = lambda s: f"{int((s > 0).sum())}-{int((s <= 0).sum())}"

print("== 1. headline re-check ==")
print(f"N={len(W)} mean {pc(W.ret)} rec {rec(W.ret)} p={sign_test(int((W.ret>0).sum()), len(W)):.4f} "
      f"worst {100*W.ret.min():.2f}% ({W.ret.idxmin()})  0.8ATR touched {(W.mae_atr<=-0.8).sum()}/26")

print("\n== 2. recency (last 12) ==")
R = W.loc[2014:, ["ret", "ret_atr", "mae_atr", "spy", "xlf", "res_x", "trv_spy"]].copy()
print((R.assign(**{c: 100 * R[c] for c in ("ret", "spy", "xlf", "res_x", "trv_spy")})).round(2).to_string())
r16 = W.loc[2016:2025]
fw = fwd_lag(cl[T], 21, 1).dropna()
base = fw.loc["2016-01-01":"2025-12-31"]
bp = float((base > 0).mean())
w16 = int((r16.ret > 0).sum())
print(f"2016-25: rec {rec(r16.ret)} mean {pc(r16.ret)} median {100*r16.ret.median():+.2f}% | own 21d base "
      f"hit {100*bp:.1f}% mean {pc(base)} | sign p vs base {sign_test(w16, len(r16), bp):.3f}")
print(f"  2016-25 resid vs XLF {pc(r16.res_x)} rec {rec(r16.res_x)}; TRV-SPY {pc(r16.trv_spy)} rec {rec(r16.trv_spy)}")
print(f"  2016-25 losers mean MAE {r16.mae_atr[r16.ret<0].mean():+.2f} ATR; all-26 losers MAE "
      f"{W.mae_atr[W.ret<0].round(2).to_dict()}")
w00 = W.loc[2000:2015]
print(f"  2000-15: rec {rec(w00.ret)} mean {pc(w00.ret)}")

print("\n== 3. mechanism (Sep conditioning) ==")
print(f"corr(TRV Sep 21d ret, window ret) = {np.corrcoef(W.sep, W.ret)[0,1]:+.2f}; "
      f"corr(Sep, TRV-SPY) = {np.corrcoef(W.sep, W.trv_spy)[0,1]:+.2f}; corr(SPY Sep, SPY window) "
      f"{np.corrcoef(W.spy_sep, W.spy)[0,1]:+.2f}")
weak = W.sep < W.sep.median()
for lbl, m in (("weak-Sep", weak), ("strong-Sep", ~weak)):
    s = W[m]
    print(f"  {lbl:10s} n={m.sum()} TRV {pc(s.ret)} ({rec(s.ret)}) SPY {pc(s.spy)} TRV-SPY {pc(s.trv_spy)} "
          f"({rec(s.trv_spy)}) resXLF {pc(s.res_x)}  yrs {list(s.index)}")
cr = [2002, 2008, 2011, 2022]
s = W[weak & ~W.index.isin(cr)]
print(f"  weak-Sep ex {cr}: n={len(s)} TRV {pc(s.ret)} TRV-SPY {pc(s.trv_spy)}")
print(f"  those 4 yrs: TRV {(100*W.loc[cr,'ret']).round(1).to_dict()} SPY {(100*W.loc[cr,'spy']).round(1).to_dict()}")
print("  peers window mean / rec, and minus SPY:")
for q in PEERS:
    s = W[q].dropna()
    print(f"    {q}: {pc(s)} {rec(s)} | ex-SPY {pc(s - W.spy[s.index])} | weak-Sep {pc(W[q][weak])} strong {pc(W[q][~weak])}")

print("\n== 4. regime overlay ==")
D = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
B = pd.read_parquet(ROOT / "data" / "market_breadth.parquet")
reg = []
for y in W.index:
    a = W.loc[y, "anchor"]
    d63 = D["63d"].loc[:a].iloc[-1] if y >= 2016 else np.nan
    nn = B.nyse_net.loc[:a].tail(5)
    reg.append({"year": y, "dial63": d63, "nyse_net5": nn.mean(), "spy_dd%": 100 * W.loc[y, "spy_dd"],
                "ret%": 100 * W.loc[y, "ret"], "trv_spy%": 100 * W.loc[y, "trv_spy"]})
G = pd.DataFrame(reg).set_index("year")
g16 = G.loc[2016:]
print(g16.round(1).to_string())
for lbl, m in (("dial63>50", g16.dial63 > 50), ("dial63<=50", g16.dial63 <= 50)):
    s = g16[m]
    print(f"  {lbl}: n={len(s)} TRV {s['ret%'].mean():+.2f}% rec {rec(s['ret%'])} yrs {list(s.index)}")
m = (G["spy_dd%"] > -2) & (G.nyse_net5 < 0)
print(f"  SPY within 2% of 52w high AND 5d NYSE net<0 (all 26): n={m.sum()} yrs {list(G.index[m])} "
      f"TRV {G.loc[m,'ret%'].round(1).to_dict()}")
m2 = G["spy_dd%"] > -2
print(f"  SPY within 2% only: n={m2.sum()} TRV mean {G.loc[m2,'ret%'].mean():+.2f}% rec {rec(G.loc[m2,'ret%'])}; "
      f"else {G.loc[~m2,'ret%'].mean():+.2f}% rec {rec(G.loc[~m2,'ret%'])}")
print(f"  today: dial main_score {D.main_score.iloc[-1]:.1f} 63d {D['63d'].iloc[-1]:.1f}; NYSE net 5d {B.nyse_net.tail(5).mean():.0f}; "
      f"SPY dd {100*(cl.SPY.iloc[-1]/cl.SPY.iloc[-252:].max()-1):.2f}%")

print("\n== 5. sizing: 113 sh, 3 ATR catastrophe stop, ATR-normalised to today (ATR 6.62) ==")
SH, A0 = 113, 6.62
pnl, mae_d = [], []
for y, p in anc.items():
    e0, ex = p + 2, p + 23
    a = atr[p - 1]
    out = (C[ex] - C[e0]) / a
    for d in range(e0 + 1, ex + 1):
        if L[d] <= C[e0] - 3 * a:
            out = min(f["Open"].values[d] - C[e0], -3 * a) / a
            break
    pnl.append(out * A0 * SH)
    mae_d.append(W.loc[y, "mae_atr"] * A0 * SH)
pnl = np.array(pnl)
nostop = W.ret_atr.values * A0 * SH
for lbl, v in (("3ATR stop", pnl), ("no stop", nostop)):
    print(f"  {lbl}: mean ${v.mean():,.0f} median ${np.median(v):,.0f} worst ${v.min():,.0f} p10 ${np.percentile(v,10):,.0f} "
          f"2016-25 mean ${v[-10:].mean():,.0f}")
print(f"  stopped yrs: {[y for y, v in zip(anc, pnl) if v <= -3 * A0 * SH + 1]}")
print(f"  worst intraday DD (no stop) ${min(mae_d):,.0f}; median ${np.median(mae_d):,.0f}; raw-% on $41k: mean ${41000*W.ret.mean():,.0f} worst ${41000*W.ret.min():,.0f}")

print("\n== 6. basket ==")
s = W.res_xk.dropna()
print(f"  resid vs XLF+KRE (2007+): n={len(s)} {pc(s)} rec {rec(s)} p={sign_test(int((s>0).sum()), len(s)):.3f}; "
      f"ex-2022 {pc(s.drop(2022, errors='ignore'))}")
s = W.res_x.drop(2022)
print(f"  resid vs XLF ex-2022: {pc(s)} rec {rec(s)} p={sign_test(int((s>0).sum()), len(s)):.3f}")
pk = W[["CB", "WRB"]].mean(axis=1)
s = W.ret - pk
print(f"  TRV minus CB/WRB avg: {pc(s)} rec {rec(s)}")
rr = cl[[T, "FORM"]].pct_change().iloc[-252:]
print(f"  1y daily corr TRV-FORM {rr.corr().iloc[0,1]:+.2f}")
