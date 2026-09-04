"""Fast triage across the 2026-08-06 candidate set. Not the falsification
pass — this only decides which candidates are worth an adversarial check.

Every cell is measured against BOTH the instrument's unconditional drift over
the same horizon and (where relevant) an all-events baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _common as C

P = C.load(["SPY", "QQQ", "XLU", "XLP", "AAPL", "GDX", "GLD", "SLV", "USO",
            "XLE", "XBI", "SMH", "IWM", "DX-Y.NYB", "^VIX", "XLK", "TLT"])
pd.set_option("display.width", 200)


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def spread_cell(a, b, cond, horizons=(3, 5, 10), label=""):
    """Forward LONG-a / SHORT-b spread return, entered at the signal close."""
    ca, cb = P[a]["Close"], P[b]["Close"]
    idx = ca.index.intersection(cb.index)
    ca, cb, cond = ca[idx], cb[idx], cond.reindex(idx).fillna(False)
    rows = []
    for k in horizons:
        sp = C.fwd(ca, k) - C.fwd(cb, k)
        sig = sp[cond & sp.notna()]
        ep = C.declusterize(sig.index)
        rows.append({**C.describe(f"{label} h{k} all-days", sig,
                                  baseline=sp.dropna()), "h": k})
        rows.append({**C.describe(f"{label} h{k} episodes", sig[ep],
                                  baseline=sp.dropna()), "h": k})
    C.show(rows)
    return cond


# --- C1  utilities washout vs SPY ------------------------------------------
hdr("C1  LONG XLU / SHORT SPY after a 21d relative washout")
xlu, spy = P["XLU"]["Close"], P["SPY"]["Close"]
rel21 = C.ret(xlu, 21) - C.ret(spy, 21)
cond = (rel21 <= rel21.rolling(756, min_periods=250).quantile(0.05)) & (C.z10(xlu) < -1.5)
print("today rel21 =", round(rel21.iloc[-1], 2), " z10(XLU) =", round(C.z10(xlu).iloc[-1], 2),
      " fires today:", bool(cond.iloc[-1]))
spread_cell("XLU", "SPY", cond, label="C1")

# --- C2  mega-cap laggard vs its index --------------------------------------
hdr("C2  LONG AAPL / SHORT QQQ when AAPL 5d rank<=5 and QQQ 5d rank>=95")
aapl, qqq = P["AAPL"]["Close"], P["QQQ"]["Close"]
ra = C.pct_rank(C.ret(aapl, 5))
rq = C.pct_rank(C.ret(qqq, 5))
cond = (ra <= 5) & (rq >= 95)
print("today AAPL r5 rank =", round(ra.iloc[-1], 1), " QQQ r5 rank =", round(rq.iloc[-1], 1),
      " fires:", bool(cond.iloc[-1]))
spread_cell("AAPL", "QQQ", cond, label="C2")

# --- C3  miners overshoot vs the metal --------------------------------------
hdr("C3  SHORT GDX / LONG GLD after miners outrun gold hard")
gdx, gld = P["GDX"]["Close"], P["GLD"]["Close"]
idx = gdx.index.intersection(gld.index)
d1 = (gdx.pct_change() - gld.pct_change()).reindex(idx) * 100
d5 = (C.ret(gdx, 5) - C.ret(gld, 5)).reindex(idx)
cond = (d1 >= 3.0) & (d5 >= 8.0)
print("today d1 =", round(d1.iloc[-1], 2), " d5 =", round(d5.iloc[-1], 2), " fires:", bool(cond.iloc[-1]))
spread_cell("GLD", "GDX", cond, label="C3 (long GLD/short GDX)")

# --- C4  crude washout vs energy equities -----------------------------------
hdr("C4  LONG USO / SHORT XLE after crude dislocates below the equities")
uso, xle = P["USO"]["Close"], P["XLE"]["Close"]
idx = uso.index.intersection(xle.index)
d5 = (C.ret(uso, 5) - C.ret(xle, 5)).reindex(idx)
r5u = C.pct_rank(C.ret(uso, 5)).reindex(idx)
cond = (d5 <= -7.0) & (r5u <= 5)
print("today d5 =", round(d5.iloc[-1], 2), " USO r5 rank =", round(r5u.iloc[-1], 1), " fires:", bool(cond.iloc[-1]))
spread_cell("USO", "XLE", cond, label="C4")

# --- C5  pre-NFP day in a stretched tape ------------------------------------
hdr("C5  SPY into NFP, conditional on a stretched 5d tape")
ev = pd.read_csv(C.ROOT / "data" / "macro_events.csv")
print(ev.columns.tolist(), len(ev))
nfp = pd.to_datetime(ev[ev.iloc[:, 1].astype(str).str.lower().str.contains("nfp|payroll")].iloc[:, 0]) \
    if ev.shape[1] >= 2 else None
print("nfp rows:", 0 if nfp is None else len(nfp))

# --- C6  melt-up + fragile dial ---------------------------------------------
hdr("C6  SPY forward after 5d rank == 100 (unconditional era view)")
r5 = C.ret(spy, 5)
rk = C.pct_rank(r5)
cond = rk >= 99.5
rows = []
for k in (3, 5, 10):
    f = C.fwd(spy, k)
    sig = f[cond & f.notna()]
    ep = C.declusterize(sig.index)
    rows.append(C.describe(f"h{k} all-days", sig, baseline=f.dropna()))
    rows.append(C.describe(f"h{k} episodes", sig[ep], baseline=f.dropna()))
C.show(rows)
print("episode era split h5:")
f5 = C.fwd(spy, 5)
sig = f5[cond & f5.notna()]
C.show(C.era_split(sig.index, sig.values))

# --- C9  biotech snapback ----------------------------------------------------
hdr("C9  LONG XBI / SHORT SPY when XBI 21d rank<=5 but 63d rank>=40 and >200d")
xbi = P["XBI"]["Close"]
r21 = C.pct_rank(C.ret(xbi, 21))
r63 = C.pct_rank(C.ret(xbi, 63))
above = xbi > xbi.rolling(200).mean()
cond = (r21 <= 5) & (r63 >= 40) & above
print("today r21 =", round(r21.iloc[-1], 1), " r63 =", round(r63.iloc[-1], 1),
      " >200d:", bool(above.iloc[-1]), " fires:", bool(cond.iloc[-1]))
spread_cell("XBI", "SPY", cond, label="C9")

# --- C12 silver vs gold ------------------------------------------------------
hdr("C12 LONG SLV / SHORT GLD after a joint up-day inside a silver 63d washout")
slv = P["SLV"]["Close"]
idx = slv.index.intersection(gld.index)
d63 = (C.ret(slv, 63) - C.ret(gld, 63)).reindex(idx)
joint = ((slv.pct_change() > 0.02) & (gld.pct_change() > 0.02)).reindex(idx).fillna(False)
cond = joint & (d63 <= -6)
print("today d63 =", round(d63.iloc[-1], 2), " joint:", bool(joint.iloc[-1]), " fires:", bool(cond.iloc[-1]))
spread_cell("SLV", "GLD", cond, label="C12")
