"""C4 round 1: oil services flush while the front contract thrusts.

Trigger: OIH pct_rank(5) <= 10 AND CL=F pct_rank(5) >= 90 (CL rank on its own
valid sessions, mapped onto OIH's calendar; live 5.2 / 92.1). Long OIH
outright and hedged (rolling-126d beta) against XLE, XOP and SPY; h=1..10,
lag=1 (MOC 09-15), explicit min_gap 10.

Also: parent attribution (OIH flush alone, crude thrust alone), neighbours
(incl. USO in place of CL=F to screen continuous-contract roll artefacts),
the C7 shock form inside this state, and the 09-14 intraday shape (gap at
the open vs an intraday slide) for OIH / SLB / XLE / XOP.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from k1_common import ASOF, atr_series, date_clusters, dial_ma10, rec  # noqa

TK = ["OIH", "XLE", "XOP", "SPY", "USO", "SLB", "HAL"]
P = load_prices(TK + ["CL=F"])
oih = P["OIH"][P["OIH"].index <= ASOF]
idx = oih.index
px = pd.DataFrame({t: P[t]["Close"].reindex(idx) for t in TK})
cl = P["CL=F"]["Close"].dropna()
cl = cl[cl.index <= ASOF]
cl_r5 = pct_rank(cl, 5).reindex(idx, method="ffill", limit=1)
cl_5d = (cl / cl.shift(5) - 1).reindex(idx, method="ffill", limit=1)
o_r5 = pct_rank(px["OIH"], 5)
uso_r5 = pct_rank(px["USO"], 5)
spy200 = px["SPY"] / px["SPY"].rolling(200).mean() - 1
dial = dial_ma10()
atr = atr_series(oih).shift(1)
o_move_atr = (px["OIH"] - px["OIH"].shift(1)) / atr
spy1 = px["SPY"].pct_change()

print(f"LIVE {ASOF.date()}: OIH r5 {o_r5.iloc[-1]:.1f}  CL r5 {cl_r5.iloc[-1]:.1f}  CL 5d {100*cl_5d.iloc[-1]:+.2f}%  "
      f"USO r5 {uso_r5.iloc[-1]:.1f}  OIH move {o_move_atr.iloc[-1]:+.2f} ATR  SPY200 {100*spy200.iloc[-1]:+.1f}%")

M = ((o_r5 <= 10) & (cl_r5 >= 90)).fillna(False)
print(f"trigger days {int(M.sum())}; first {M[M].index[0].date()} last {M[M].index[-1].date()}")

variants = {
    "OIH r5<=5 & CL>=90": ((o_r5 <= 5) & (cl_r5 >= 90)).fillna(False),
    "OIH r5<=15 & CL>=90": ((o_r5 <= 15) & (cl_r5 >= 90)).fillna(False),
    "OIH r5<=20 & CL>=90": ((o_r5 <= 20) & (cl_r5 >= 90)).fillna(False),
    "OIH r5<=10 & CL>=80": ((o_r5 <= 10) & (cl_r5 >= 80)).fillna(False),
    "OIH r5<=10 & CL>=95": ((o_r5 <= 10) & (cl_r5 >= 95)).fillna(False),
    "OIH r5<=10 & USO r5>=90": ((o_r5 <= 10) & (uso_r5 >= 90)).fillna(False),
    "OIH r5<=10 & CL 5d>=+8%": ((o_r5 <= 10) & (cl_5d >= 0.08)).fillna(False),
    "PARENT OIH r5<=10 alone": (o_r5 <= 10).fillna(False),
    "PARENT CL r5>=90 alone": (cl_r5 >= 90).fillna(False),
    "OIH r5<=10 & CL r5<=50 (crude NOT up)": ((o_r5 <= 10) & (cl_r5 <= 50)).fillna(False),
}
battery(px, M, [("OIH", 1.0)], 5, "C4 LONG OIH outright", cost_bps=3.0,
        variants=variants, min_gap=10, event_kinds=("fomc_decision",))
battery(px, M, [("OIH", 1.0)], 10, "C4 LONG OIH outright", cost_bps=3.0,
        variants=None, min_gap=10, event_kinds=("fomc_decision",))

# hedged vehicles with rolling beta
r1 = px.pct_change()


def rbeta(a, b):
    return r1[a].rolling(126).cov(r1[b]) / r1[b].rolling(126).var()


def fwdl(t, h):
    return fwd_lag(px[t], h, 1)


trig = idx[M.values]
rows = []
for hedge in ("XLE", "XOP", "SPY"):
    b = rbeta("OIH", hedge)
    for h in (1, 2, 3, 5, 10):
        res = fwdl("OIH", h) - b * fwdl(hedge, h)
        eq = fwdl("OIH", h) - fwdl(hedge, h)
        valid = res.dropna().index
        t_ = trig.intersection(valid)
        if len(t_) == 0:
            continue
        ep = declusters(t_, 10, valid)
        v = res.loc[ep].values
        span = valid[(valid >= ep[0])]
        s = summarize(v, f"OIH-beta*{hedge} h={h}")
        s["ctrl_span_pct"] = 100 * res.loc[span].mean()
        s["eqdollar_pct"] = 100 * eq.loc[ep].mean()
        s["hedge_leg_pct"] = 100 * fwdl(hedge, h).loc[ep].mean()
        s["beta_med"] = float(np.nanmedian(b.loc[ep]))
        s["rec"] = rec(v)
        rows.append(s)
show(rows, "HEDGED vehicles (episodes, gap 10)")

h = 5
f5 = fwdl("OIH", h)
valid = f5.dropna().index
ep = declusters(trig.intersection(valid), 10, valid)
E = pd.DataFrame({"d": ep, "oih5": f5.loc[ep].values, "spy200": spy200.loc[ep].values,
                  "dial": dial.reindex(ep).values, "oih_atr": o_move_atr.loc[ep].values,
                  "cl5d": cl_5d.loc[ep].values, "cl_r5": cl_r5.loc[ep].values,
                  "o_r5": o_r5.loc[ep].values,
                  "gapshare": ((oih["Open"] - oih["Close"].shift(1)) /
                               (oih["Close"] - oih["Close"].shift(1))).loc[ep].values})
bx = rbeta("OIH", "XLE")
E["res_xle5"] = (f5 - bx * fwdl("XLE", h)).loc[ep].values
print("\nEPISODES (h=5): date, OIH fwd5 %, residual vs XLE %, OIH 1d ATR move, CL 5d %, SPY vs 200d %, dial")
print(E.assign(oih5=100 * E.oih5, res_xle5=100 * E.res_xle5, cl5d=100 * E.cl5d, spy200=100 * E.spy200)
      .round(2).to_string(index=False))


def line(sub, col, label):
    s = summarize(sub[col].values, label)
    s["rec"] = rec(sub[col].values)
    return s


show([line(E, "oih5", "all"), line(E[E.d < "2018-01-01"], "oih5", "pre-2018"),
      line(E[E.d >= "2018-01-01"], "oih5", "2018+"),
      line(E[E.spy200 >= 0], "oih5", "SPY>=200d"), line(E[E.spy200 < 0], "oih5", "SPY<200d"),
      line(E[E.oih_atr <= -1.5], "oih5", "flush day itself <= -1.5 ATR"),
      line(E[E.oih_atr > -1.5], "oih5", "flush day > -1.5 ATR"),
      line(E, "res_xle5", "residual vs XLE all"),
      line(E[E.d >= "2018-01-01"], "res_xle5", "residual vs XLE 2018+"),
      line(E[E.spy200 >= 0], "res_xle5", "residual vs XLE SPY>=200d")], "splits, h=5 episodes")

# C7 shock form inside the state
shockM = ((o_move_atr <= -1.5) & (spy1 > -0.0075)).fillna(False)
for lbl, mm in (("OIH calm shock & CL r5>=90", shockM & (cl_r5 >= 90)),
                ("OIH calm shock & CL r5<90", shockM & ~(cl_r5 >= 90))):
    t_ = idx[mm.fillna(False).values]
    rr = []
    for hh in (1, 3, 5):
        res = fwdl("OIH", hh) - rbeta("OIH", "SPY") * fwdl("SPY", hh)
        vv = res.dropna().index
        e = declusters(t_.intersection(vv), 5, vv)
        s = summarize(res.loc[e].values, f"{lbl} LONG resid vs SPY h={hh}")
        s["rec"] = rec(res.loc[e].values)
        rr.append(s)
    show(rr)
    print("   dates:", [str(d.date()) for d in declusters(t_, 5, idx)])

# intraday shape on 2026-09-14
print("\n=== 09-14 INTRADAY SHAPE (15min, ET) ===")
try:
    from intraday_data import get_intraday_for_date
    for t in ("OIH", "SLB", "XLE", "XOP", "HAL", "SPY"):
        b = get_intraday_for_date(t, "2026-09-14")
        prev = P[t]["Close"].loc[:"2026-09-11"].iloc[-1]
        if b is None or len(b) == 0:
            print(f"  {t}: no bars")
            continue
        o, c = b["open"].iloc[0], b["close"].iloc[-1]
        lo = b["low"].min()
        tot = c / prev - 1
        gap = o / prev - 1
        b = b.copy()
        b["cum"] = b["close"] / prev - 1
        at1030 = b[b["ts"].astype(str).str.contains("10:15|10:30")]["cum"]
        print(f"  {t}: prev close {prev:.2f}  open gap {100*gap:+.2f}%  close {100*tot:+.2f}%  "
              f"gap share {gap/tot if tot else float('nan'):.2f}  low {100*(lo/prev-1):+.2f}%  bars {len(b)}  "
              f"first ts {b['ts'].iloc[0]}  last {b['ts'].iloc[-1]}")
        print("     cum path every hour:", [f"{str(r.ts)[11:16]} {100*r.cum:+.2f}" for r in b.iloc[::4].itertuples()])
except Exception as exc:  # noqa
    print("  intraday unavailable:", exc)
print("\n  daily-bar gap check 09-14:")
for t in ("OIH", "SLB", "XLE", "XOP", "HAL"):
    d = P[t]
    o, c, pc = d["Open"].loc["2026-09-14"], d["Close"].loc["2026-09-14"], d["Close"].loc["2026-09-11"]
    print(f"   {t}: gap {100*(o/pc-1):+.2f}%  open->close {100*(c/o-1):+.2f}%  day {100*(c/pc-1):+.2f}%")
