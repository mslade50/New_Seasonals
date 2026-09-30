"""C1 round 1: long gold after a one-day crash of >= 3.5% (about 2 ATR), with the
dollar and the ten-year at 252 highs.

Parent: GLD 1d return <= -3.5%. Interaction gate: UUP (or DX) and ^TNX at/near
their 252 highs. Vehicles GLD (2004+) and GC=F (2000+, crash defined on GC=F's
own return, seams checked against GLD). Entry lag 1 (MOC next close).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-28")
TK = ["GLD", "GC=F", "UUP", "DX-Y.NYB", "^TNX", "SLV", "GDX", "XME", "SPY"]
raw = load_prices(TK)


def on_index(idx, t):
    return raw[t]["Close"].reindex(idx).ffill(limit=3)


def build(anchor: str):
    idx = raw[anchor].loc[:BAR].index
    px = pd.DataFrame({t: on_index(idx, t) for t in TK})
    a = raw[anchor].loc[:BAR]
    atr = pd.Series(wilder_atr(a["High"], a["Low"], a["Close"]), index=idx)
    r1 = px[anchor].pct_change(fill_method=None)
    ratr = (px[anchor] - px[anchor].shift(1)) / atr.shift(1)
    hi = lambda s: s.rolling(252, min_periods=200).max()
    st = pd.DataFrame(index=idx)
    st["r1"] = r1
    st["ratr"] = ratr
    st["uup_hi"] = px["UUP"] >= hi(px["UUP"]) - 1e-9
    st["dx_hi"] = px["DX-Y.NYB"] >= hi(px["DX-Y.NYB"]) - 1e-9
    st["tnx_hi"] = px["^TNX"] >= hi(px["^TNX"]) - 1e-9
    # "near" forms: within 1% (dollar) / 3% of level (yield), or a 252 high in last 5 sessions
    st["dx_near"] = px["DX-Y.NYB"] >= 0.99 * hi(px["DX-Y.NYB"])
    st["tnx_near"] = px["^TNX"] >= 0.97 * hi(px["^TNX"])
    st["dx_hi5"] = st["dx_hi"].rolling(5).max().astype(bool)
    st["tnx_hi5"] = st["tnx_hi"].rolling(5).max().astype(bool)
    st["gl_off_hi"] = px[anchor] / hi(px[anchor]) - 1
    st["gl_200"] = px[anchor] / px[anchor].rolling(200).mean() - 1
    return px, st


for anchor in ("GLD", "GC=F"):
    px, st = build(anchor)
    print(f"\n\n################ ANCHOR {anchor} ################")
    last = st.loc[BAR]
    print("TODAY:", {k: (round(float(v), 4) if not isinstance(v, (bool, np.bool_)) else bool(v))
                     for k, v in last.items()})
    crash = st["r1"] <= -0.035
    prior21 = crash.shift(1).rolling(21, min_periods=1).max().fillna(0).astype(bool)
    first = crash & ~prior21
    dollar_rates = st["dx_near"] & st["tnx_near"]
    strict = (st["uup_hi"] | st["dx_hi"]) & st["tnx_hi"]
    print(f"crash days {int(crash.sum())}, first-in-21d {int(first.sum())}, "
          f"crash&strict(dollar hi & tnx hi) {int((crash & strict).sum())}, "
          f"crash&near {int((crash & dollar_rates).sum())}, "
          f"crash&dx_hi5&tnx_hi5 {int((crash & st['dx_hi5'] & st['tnx_hi5']).sum())}")
    print("crash days prior-21d count today:", bool(prior21.loc[BAR]))
    variants = {
        "r1<=-3.0%": st["r1"] <= -0.030,
        "r1<=-3.5% (parent)": crash,
        "r1<=-4.0%": st["r1"] <= -0.040,
        "ATR<=-1.5": st["ratr"] <= -1.5,
        "ATR<=-2.0": st["ratr"] <= -2.0,
        "ATR<=-2.5": st["ratr"] <= -2.5,
        "first crash in 21d": first,
        "repeat crash in 21d": crash & prior21,
        "crash & DX near hi & TNX near hi": crash & dollar_rates,
        "crash & NOT(dollar_rates near)": crash & ~dollar_rates,
        "crash & DX near hi": crash & st["dx_near"],
        "crash & TNX near hi": crash & st["tnx_near"],
        "crash & dx_hi5 & tnx_hi5": crash & st["dx_hi5"] & st["tnx_hi5"],
        "crash & GLD<200d": crash & (st["gl_200"] < 0),
        "crash & GLD>=200d": crash & (st["gl_200"] >= 0),
        "crash & >=15% off hi": crash & (st["gl_off_hi"] <= -0.15),
    }
    for h in (1, 3, 5):
        battery(px, crash.loc[:BAR], [(anchor, 1.0)], h,
                f"{anchor} long after {anchor} <= -3.5% day", cost_bps=3 if anchor == "GLD" else 1,
                variants=variants, event_kinds=("nfp",))

    # gated cell episode list with h=1..5
    for lbl, m in (("crash & DX near & TNX near", crash & dollar_rates),
                   ("crash & strict hi", crash & strict),
                   ("crash & dx_hi5 & tnx_hi5", crash & st["dx_hi5"] & st["tnx_hi5"])):
        d = px.index[m.fillna(False).values]
        rows = []
        for dd in d:
            row = {"date": dd.date(), "r1%": round(100 * st.loc[dd, "r1"], 2),
                   "atr": round(st.loc[dd, "ratr"], 2),
                   "offhi%": round(100 * st.loc[dd, "gl_off_hi"], 1),
                   "v200%": round(100 * st.loc[dd, "gl_200"], 1)}
            for h in (1, 2, 3, 5):
                v = fwd_lag(px[anchor], h, 1).get(dd, np.nan)
                row[f"h{h}%"] = round(100 * v, 2) if pd.notna(v) else np.nan
            rows.append(row)
        print(f"\n--- {anchor}: {lbl} episodes (lag1) ---")
        print(pd.DataFrame(rows).to_string(index=False) if rows else "  none")

# book overlap: XME on GLD crash episodes and trailing correlation
px, st = build("GLD")
crash = st["r1"] <= -0.035
d = declusters(px.index[crash.fillna(False).values], 3, px.index)
for h in (1, 3, 5):
    g = fwd_lag(px["GLD"], h).reindex(d)
    x = fwd_lag(px["XME"], h).reindex(d)
    ok = g.notna() & x.notna()
    print(f"\noverlap h={h}: GLD {100*g[ok].mean():+.3f}%  XME {100*x[ok].mean():+.3f}%  "
          f"corr {np.corrcoef(g[ok], x[ok])[0,1]:+.2f}  n={int(ok.sum())}")
dr = px[["GLD", "XME"]].pct_change(fill_method=None).dropna().iloc[-252:]
print(f"trailing-252 daily corr GLD/XME {dr.corr().iloc[0,1]:+.3f}")
