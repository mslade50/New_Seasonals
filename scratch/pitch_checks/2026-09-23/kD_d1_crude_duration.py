"""kD D1 round 1: long IEF (TLT neighbour) after a crude collapse
(USO 5d rank <= 3) while ^TNX 63d rank >= 80, h=5. Pre-specified sign LONG
duration. Job: kill it.

Mirror of watchlist 40 (short IEF, commodity complex at 252 high + print in
hold). Construction reused from 2026-09-08/c8b_print_gate_and_regime.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["IEF", "TLT", "USO", "CL=F", "^TNX", "DBC", "SPY"]
px = close_panel(TK)
raw = load_prices(TK)
cl = px["CL=F"].copy()
cl[cl <= 0] = np.nan          # April 2020 negative print breaks returns
px["CL=F"] = cl
IDX = px.index

r5_uso = pct_rank(px["USO"], 5)
r5_cl = pct_rank(px["CL=F"], 5)
r63_tnx = pct_rank(px["^TNX"], 63)
r5_ief = pct_rank(px["IEF"], 5)
r5_tlt = pct_rank(px["TLT"], 5)
r5_dbc = pct_rank(px["DBC"], 5)

print("LIVE (last row of each series):")
for nm, s in (("USO r5", r5_uso), ("CL=F r5", r5_cl), ("TNX r63", r63_tnx),
              ("IEF r5", r5_ief), ("TLT r5", r5_tlt), ("DBC r5", r5_dbc)):
    v = s.dropna()
    print(f"  {nm:8s} {v.index[-1].date()} {v.iloc[-1]:.1f}")
for t in ("USO", "CL=F", "IEF", "TLT", "^TNX"):
    s = px[t].dropna()
    print(f"  {t} 5d ret {100*(s.iloc[-1]/s.iloc[-6]-1):+.2f}%  close {s.iloc[-1]:.3f}")
for t in ("IEF", "TLT"):
    df = raw[t]
    a = wilder_atr(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy())
    print(f"  {t} Wilder-14 ATR {a[-1]:.4f}  ({100*a[-1]/df['Close'].iloc[-1]:.2f}% of close)")

crude = r5_uso <= 3
rates = r63_tnx >= 80
cell = (crude & rates).fillna(False)
cell_cl = ((r5_cl <= 3) & rates).fillna(False)

variants = {
    "USO r5<=2 & TNX>=80": (r5_uso <= 2) & rates,
    "USO r5<=5 & TNX>=80": (r5_uso <= 5) & rates,
    "USO r5<=10 & TNX>=80": (r5_uso <= 10) & rates,
    "CL=F r5<=3 & TNX>=80": cell_cl,
    "USO r5<=3 & TNX>=70": crude & (r63_tnx >= 70),
    "USO r5<=3 & TNX>=90": crude & (r63_tnx >= 90),
    "CTRL rates only, NO crude (TNX>=80 & USO r5>3)": rates & (r5_uso > 3),
    "CTRL crude only, NO rates (USO r5<=3 & TNX<80)": crude & (r63_tnx < 80),
    "crude only (USO r5<=3, any rates)": crude,
    "rates only (TNX>=80, any crude)": rates,
}

H = 5
for legs, nm in (([("IEF", 1.0)], "LONG IEF"), ([("TLT", 1.0)], "LONG TLT")):
    battery(px, cell, legs, H, f"D1 {nm}: USO r5<=3 & TNX r63>=80", cost_bps=2.5,
            variants=variants, event_kinds=("cpi", "ppi"))
    ret = vehicle_ret(px, legs, H, 1)
    valid = ret.notna()
    dd = IDX[cell.values & valid.values]
    epi = declusters(dd, H, IDX)
    ep = ret.loc[epi].values
    # NFP tail split
    fl = event_in_window(epi, IDX, H, 1, ("nfp",))
    show([summarize(ep[fl], f"NFP IN h=5 hold (N={int(fl.sum())})"),
          summarize(ep[~fl], f"NFP OUT (N={int((~fl).sum())})")], f"{nm} NFP split")
    # no-print slice honestly (today's state: no CPI/PPI in the hold)
    fp = event_in_window(epi, IDX, H, 1, ("cpi", "ppi"))
    fn = event_in_window(epi, IDX, H, 1, ("nfp",))
    live_like = ~fp & ~fn
    show([summarize(ep[~fp], "no CPI/PPI in hold"),
          summarize(ep[live_like], "no CPI/PPI/NFP in hold (LIVE-LIKE)")],
         f"{nm} live calendar slice")
    # year splits
    yrs = pd.DatetimeIndex(epi).year
    show([summarize(ep[yrs == 2022], "2022 only"),
          summarize(ep[yrs != 2022], "ex-2022"),
          summarize(ep[yrs % 4 == 2], "midterm yrs"),
          summarize(ep[yrs % 4 != 2], "non-midterm"),
          summarize(ep[pd.DatetimeIndex(epi) < "2020-05-01"], "pre USO 2020-05 reset"),
          summarize(ep[pd.DatetimeIndex(epi) >= "2020-05-01"], "post 2020-05")],
         f"{nm} era / regime splits (episodes)")
    # IEF own 5d prior return in the episodes (is the lag already spent?)
    own5 = r5_ief.reindex(epi).values
    hi_own = own5 >= 60
    show([summarize(ep[hi_own], f"IEF r5>=60 already (live-like, N={int(hi_own.sum())})"),
          summarize(ep[~hi_own & ~np.isnan(own5)], "IEF r5<60")],
         f"{nm} split on IEF's own prior 5d rank (live IEF r5 77)")
    print("  episode table:")
    for d, v, o, u, t in zip(epi, ep, own5, r5_uso.reindex(epi).values,
                             r63_tnx.reindex(epi).values):
        print(f"   {d.date()}  ret {100*v:+.2f}%  IEF r5 {o:5.1f}  USO r5 {u:4.1f}  TNX r63 {t:5.1f}")
