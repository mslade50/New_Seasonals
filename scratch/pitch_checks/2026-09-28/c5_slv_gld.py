"""C5 round 1.
A) Long SLV against beta-GLD when the gold/silver ratio sits at a trailing-252
   extreme (level pct rank >= 95) after a >= 35% silver drawdown. h=5..10.
   Run on ETFs (SLV/GLD, PIT trailing-252 beta) and on futures (SI=F/GC=F).
B) Long GLD at h=5/10 when GLD is >= 15% under its 252 high and DX 21d rank >= 85.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd


def eps(ret, mask, h, idx, gap=None):
    s = idx[mask.reindex(idx, fill_value=False).values & ret.notna().values]
    e = declusters(s, gap or h, idx)
    return e, ret.loc[e].values


def row(v, lbl, ctrl=None):
    r = summarize(v, lbl)
    if len(v):
        r["sign_p"] = sign_test(int((np.asarray(v) > 0).sum()), len(v))
    if ctrl is not None:
        r["excess_pp"] = round(r["mean_pct"] - 100 * ctrl, 3)
    return r


def pair_block(s_t: str, g_t: str, cost: float):
    px = close_panel([s_t, g_t]).dropna()
    idx = px.index
    s, g = px[s_t], px[g_t]
    ratio = g / s
    rr = ratio.rolling(252).rank(pct=True) * 100
    s_dd = 100 * (1 - s / s.rolling(252).max())
    g_dd = 100 * (1 - g / g.rolling(252).max())
    rs, rg = s.pct_change(), g.pct_change()
    beta = rs.rolling(252).cov(rg) / rg.rolling(252).var()
    print(f"\n######## {s_t}/{g_t}: LIVE {idx[-1].date()} ratio rank252 {rr.iloc[-1]:.1f} "
          f"(252max? {ratio.iloc[-1] >= ratio.rolling(252).max().iloc[-1] - 1e-12}) "
          f"{s_t} dd {s_dd.iloc[-1]:.2f}% {g_t} dd {g_dd.iloc[-1]:.2f}% beta252 {beta.iloc[-1]:.3f}")
    print("ratio rank last 8:", rr.tail(8).round(1).tolist())
    gate = (rr >= 95) & (s_dd >= 35)
    masks = {"PRE-SPEC rr>=95 & sdd>=35": gate,
             "rr>=90 & sdd>=35": (rr >= 90) & (s_dd >= 35),
             "rr>=98 & sdd>=35": (rr >= 98) & (s_dd >= 35),
             "rr>=95 & sdd>=30": (rr >= 95) & (s_dd >= 30),
             "rr>=95 & sdd>=40": (rr >= 95) & (s_dd >= 40),
             "rr>=95 alone (no dd gate)": rr >= 95,
             "sdd>=35 alone (no ratio gate)": s_dd >= 35,
             "sdd>=35 & rr<95 (complement)": (s_dd >= 35) & (rr < 95)}
    for h in (5, 10):
        rpair = fwd_lag(s, h) - beta * fwd_lag(g, h)
        rslv = fwd_lag(s, h)
        rows = []
        for lbl, m in masks.items():
            e, v = eps(rpair, m, h, idx)
            rows.append(row(v, f"pair {lbl} (N={len(e)})", rpair.mean()))
        e, v = eps(rslv, gate, h, idx)
        rows.append(row(v, f"long {s_t} outright PRE-SPEC (N={len(e)})", rslv.mean()))
        show(rows, f"{s_t}-beta*{g_t} h={h} (excess vs all-days pair drift "
                   f"{100*rpair.mean():+.3f}%, {s_t} drift {100*rslv.mean():+.3f}%)")
        e, v = eps(rpair, gate, h, idx)
        if len(e):
            show(era_split(e, v), f"era split pair h={h}")
            print("  ", cluster_note(e, v))
            loc = local_control(idx[rpair.notna().values], idx[gate.reindex(idx, fill_value=False).values])
            print(f"   local +/-126 ctrl {100*rpair.loc[loc].mean():+.3f}%  "
                  f"cost {cost*2:.0f}bp -> {100*100*np.nanmean(v)/(cost*2):.1f}x")
            print("   episodes:", ", ".join(f"{d.date()} {100*x:+.1f}" for d, x in zip(e, v)))


pair_block("SLV", "GLD", 3.0)
pair_block("SI=F", "GC=F", 1.5)

# ---------- B: long GLD in a drawdown with the dollar thrusting ----------
gp = close_panel(["GLD"]).dropna()
dx = close_panel(["DX-Y.NYB"])["DX-Y.NYB"].dropna()
dxr = pct_rank(dx, 21).reindex(gp.index, method="ffill", limit=2)
g = gp["GLD"]
gdd = 100 * (1 - g / g.rolling(252).max())
print(f"\n######## B LIVE: GLD dd {gdd.iloc[-1]:.2f}%  DX r21 {dxr.iloc[-1]:.1f}")
gateB = (gdd >= 15) & (dxr >= 85)
varB = {"dd>=10 & dx>=85": (gdd >= 10) & (dxr >= 85), "dd>=20 & dx>=85": (gdd >= 20) & (dxr >= 85),
        "dd>=15 & dx>=75": (gdd >= 15) & (dxr >= 75), "dd>=15 & dx>=95": (gdd >= 15) & (dxr >= 95),
        "dd>=15 alone (no DX gate)": gdd >= 15, "dx>=85 alone (no dd gate)": dxr >= 85,
        "dd>=15 & dx<85 (complement)": (gdd >= 15) & (dxr < 85)}
for h in (5, 10):
    battery(gp, gateB, [("GLD", 1.0)], h, "C5B long GLD dd>=15 & DX r21>=85",
            cost_bps=3.0, variants=varB, event_kinds=("nfp",))
    ret = fwd_lag(g, h)
    e, v = eps(ret, gateB, h, gp.index)
    print(f"  sign test {int((v>0).sum())}-{int((v<=0).sum())} p={sign_test(int((v>0).sum()), len(v)):.4f}")
