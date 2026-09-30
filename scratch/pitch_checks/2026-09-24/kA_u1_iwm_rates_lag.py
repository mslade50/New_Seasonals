"""kA U1 round 1: long IWM against beta-SPY the session after IWM lags beta-SPY
by >= 0.8pp on a >= +10 bp ^TNX day, h=1..5. Pre-specified sign LONG residual
(reversal). Ex-ante beta: trailing 63d OLS on daily returns through t-1.
battery() takes fixed weights, so the time-varying-beta spread is scored with
the same pitch_lab primitives in bat() below; battery() on a fixed beta is run
as the cross-check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["IWM", "SPY", "^TNX"]
px = close_panel(TK)
px = px[px["IWM"].notna() & px["SPY"].notna()]
IDX = px.index
ri = px["IWM"].pct_change()
rs = px["SPY"].pct_change()
tnx = px["^TNX"].ffill()
dtnx = tnx - tnx.shift(1)


def roll_beta(n: int) -> pd.Series:
    cov = ri.rolling(n).cov(rs)
    var = rs.rolling(n).var()
    return (cov / var).shift(1)          # through t-1 (ex-ante)


beta63 = roll_beta(63)
beta252 = roll_beta(252)
resid63 = ri - beta63 * rs
resid252 = ri - beta252 * rs

print("LIVE 2026-09-23:")
print(f"  IWM {100*ri.iloc[-1]:+.2f}%  SPY {100*rs.iloc[-1]:+.2f}%  beta63 {beta63.iloc[-1]:.3f}"
      f"  beta252 {beta252.iloc[-1]:.3f}  resid63 {100*resid63.iloc[-1]:+.2f}pp"
      f"  resid252 {100*resid252.iloc[-1]:+.2f}pp  dTNX {100*dtnx.iloc[-1]:+.1f} bp")
raw = load_prices(["IWM", "SPY"])
for t in ("IWM", "SPY"):
    df = raw[t]
    a = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
    print(f"  {t} close {df['Close'].iloc[-1]:.2f} Wilder-14 ATR {a[-1]:.3f} "
          f"({100*a[-1]/df['Close'].iloc[-1]:.2f}%)")


def spread_ret(h: int, beta: pd.Series, lag: int = 1) -> pd.Series:
    return fwd_lag(px["IWM"], h, lag) - beta * fwd_lag(px["SPY"], h, lag)


def bat(mask: pd.Series, h: int, beta: pd.Series, title: str, cost_bps: float = 4.5,
        variants: dict | None = None) -> None:
    ret = spread_ret(h, beta)
    naive = spread_ret(h, beta, lag=0)
    valid = ret.notna()
    sig = IDX[mask.reindex(IDX, fill_value=False).values & valid.values]
    span = (sig[0], sig[-1])
    in_span = (IDX >= span[0]) & (IDX <= span[1]) & valid.values
    epi = declusters(sig, h, IDX)
    ep = ret.loc[epi].values
    loc = local_control(IDX[valid.values], sig)
    print(f"\n##### {title} (h={h}, lag=1) span {span[0].date()}..{span[1].date()} #####")
    show([summarize(ret.loc[sig].values, f"COND day-level (N={len(sig)})"),
          summarize(ep, f"COND episodes (N={len(epi)})"),
          summarize(ret[in_span].values, "CTRL-a own drift, same span"),
          summarize(ret[valid].values, "CTRL-b all days"),
          summarize(ret.loc[loc].values, "CTRL-c local +/-126td"),
          summarize(naive.loc[sig].values, "COND lag=0 (no-lag)")], "1. cond vs controls")
    ctrl = ret[in_span].values
    se = np.sqrt(ep.var(ddof=1) / len(ep) + ctrl.var(ddof=1) / len(ctrl))
    w = int((ep > 0).sum())
    print(f"  diff {100*(ep.mean()-ctrl.mean()):+.3f}%  welch t {(ep.mean()-ctrl.mean())/se:+.2f}"
          f"  boot P<=0 {bootstrap_p_le0(ep):.3f}  record {w}-{len(ep)-w} sign p {sign_test(w, len(ep)):.4f}")
    print(f"  concentration: {cluster_note(epi, ep)}")
    show(era_split(epi, ep), "2+3. episode era split")
    mid = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(ep[mid], "midterm"), summarize(ep[~mid], "non-midterm")], "midterm split")
    print(f"  worst episode {100*ep.min():.2f}% on {epi[int(np.argmin(ep))].date()}")
    if variants:
        vr = []
        for lbl, m in variants.items():
            s = IDX[m.reindex(IDX, fill_value=False).values & valid.values]
            if len(s) == 0:
                vr.append({"label": lbl, "n": 0})
                continue
            e = declusters(s, h, IDX)
            r = summarize(ret.loc[e].values, lbl)
            r["n_days"] = len(s)
            vr.append(r)
        show(vr, "4. threshold / gate attribution (episodes)")
    print(f"5. cost: pair ~{cost_bps} bps; episode mean {1e4*ep.mean():.1f} bps -> "
          f"{1e4*ep.mean()/cost_bps:.1f}x")
    fl = event_in_window(epi, IDX, h, 1, ("nfp", "cpi"))
    show([summarize(ep[fl], f"nfp/cpi IN (N={int(fl.sum())})"),
          summarize(ep[~fl], f"nfp/cpi OUT (N={int((~fl).sum())})")], "6. print in hold")


lag_ = resid63 <= -0.008
rate = dtnx >= 0.10
cell = (lag_ & rate).fillna(False)
print(f"\ncell days {int(cell.sum())}  (lag-any-day {int(lag_.sum())}, rate days {int(rate.sum())})")

variants = {
    "resid<=-0.6 & dTNX>=10": (resid63 <= -0.006) & rate,
    "resid<=-1.0 & dTNX>=10": (resid63 <= -0.010) & rate,
    "resid<=-0.8 & dTNX>=7": lag_ & (dtnx >= 0.07),
    "resid<=-0.8 & dTNX>=13": lag_ & (dtnx >= 0.13),
    "beta252 resid<=-0.8 & dTNX>=10": (resid252 <= -0.008) & rate,
    "GATE OFF: resid<=-0.8 any day": lag_,
    "complement: resid<=-0.8 & dTNX<10": lag_ & (dtnx < 0.10),
    "resid<=-0.8 & dTNX<=-10 (rates DOWN)": lag_ & (dtnx <= -0.10),
    "rates day only (dTNX>=10, any resid)": rate,
    "resid<=-0.8 & dTNX>=10 & SPY<0": cell & (rs < 0),
    "resid<=-0.8 & dTNX>=10 & SPY>=0": cell & (rs >= 0),
}
for H in (1, 2, 3, 5):
    bat(cell, H, beta63, "U1 long IWM - beta63*SPY: resid63<=-0.8pp & dTNX>=+10bp",
        variants=variants if H in (1, 5) else None)

# cross-check with pitch_lab.battery on today's fixed beta
b_now = float(beta63.iloc[-1])
battery(px, cell, [("IWM", 1.0), ("SPY", -b_now)], 5,
        f"U1 cross-check fixed beta {b_now:.2f}", cost_bps=2.25, event_kinds=("nfp", "cpi"))
