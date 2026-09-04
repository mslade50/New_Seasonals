"""B1 recon, second enumeration: every live tape extreme, by asset class.

One cheap pass per state so the surface map can give each cell a NUMBER
instead of an adjective.  Everything is lag=1 MOC entry, declustered at 5 td,
and scored against the instrument's OWN drift over the same horizon (the
2026-08-07 lesson: the control is never zero).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, fwd_lag, declusters, summarize, sign_test, pct_rank  # noqa: E402

warnings.filterwarnings("ignore")

TK = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "GDX",
      "USO", "DBC", "UUP", "DX-Y.NYB", "EFA", "EEM", "FXI", "SVXY",
      "XLU", "XLV", "XLP", "XLRE", "VNQ", "SMH", "XLE", "XLK", "XLF", "IBB"]
px = close_panel(TK)
idx = px.index
HS = (1, 2, 3, 5, 10)


# BUGFIX 2026-08-11 (caught by the C4/C8/C11 checker, a4_c4_c11_teardown.py):
# pitch_lab.pct_rank takes a PRICE series and differences it internally, so the
# original `rank(r(px[T], n), n)` here ranked pct_change(n) OF pct_change(n) --
# a second difference on a series that crosses zero constantly. Every rank
# state below (1, 3, 4, 5, 7, 8) was measuring a statistic nobody meant to
# compute; the 52w-extreme state (2) and the USO 1d state (6) were never
# affected because they never went through pct_rank. Feed pct_rank the PRICE.
def rank(s, n):
    return pct_rank(s, n)


def r(s, n):
    return s.pct_change(n)


def score(name, mask_ser, tkr, note=""):
    s = px[tkr].dropna()
    m = mask_ser.reindex(s.index).fillna(False)
    trig = declusters(s.index[m.values], 5, s.index)
    if len(trig) < 4:
        print(f"{name:<44} {tkr:<9} N={len(trig)}  -- too few, cell is rare")
        return
    out = [f"{name:<44} {tkr:<9} N={len(trig):<4}"]
    for h in HS:
        f = fwd_lag(s, h, lag=1)
        v = f.reindex(trig).dropna()
        if len(v) < 4:
            continue
        st = summarize(v.values)
        own = f.dropna().mean() * 100
        p = sign_test(int((v.values > 0).sum()), len(v))
        out.append(f"  h{h}: {st['mean_pct']:+6.2f} (own {own:+5.2f}, ex {st['mean_pct']-own:+6.2f}, "
                   f"hit {st['hit']:4.1f}, p {p:.3f})")
    print("\n".join(out) + (f"\n    {note}" if note else ""))
    print()


print("=" * 100)
print("STATE 1  utilities / bond-proxy washout (XLU 21d rank <= 5)")
print("=" * 100)
xlu_rank21 = rank(px["XLU"], 21)
score("XLU 21d rank <= 5", xlu_rank21 <= 5, "XLU")
score("XLU 21d rank <= 5", xlu_rank21 <= 5, "VNQ")
score("XLU 21d rank <= 5", xlu_rank21 <= 5, "XLP")

print("=" * 100)
print("STATE 2  SPY at a 52w high WHILE TLT is at a 52w low (duration divergence)")
print("=" * 100)
spy_hi = px["SPY"] >= px["SPY"].rolling(252).max() * 0.999
tlt_lo = px["TLT"] <= px["TLT"].rolling(252).min() * 1.01
div = spy_hi & tlt_lo
print(f"joint state days: {int(div.sum())}, declustered: {len(declusters(idx[div.fillna(False).values], 5, idx))}")
for t in ["SPY", "TLT", "XLU", "IWM", "XLV"]:
    score("SPY 52wh & TLT within 1% of 52wl", div, t)

print("=" * 100)
print("STATE 3  precious-metals thrust (GDX 5d rank >= 95)  [GDX pitched 2026-08-10]")
print("=" * 100)
gdx_thrust = rank(px["GDX"], 5) >= 95
for t in ["GDX", "GLD", "SLV", "SPY"]:
    score("GDX 5d rank >= 95", gdx_thrust, t)

print("=" * 100)
print("STATE 4  healthcare leadership at a 52w high (XLV 63d rank >= 98)")
print("=" * 100)
xlv_lead = rank(px["XLV"], 63) >= 98
for t in ["XLV", "IBB", "SPY"]:
    score("XLV 63d rank >= 98", xlv_lead, t)

print("=" * 100)
print("STATE 5  EM laggard vs China leader (EEM 63d rank <= 5 & FXI 21d rank >= 90)")
print("=" * 100)
em_div = (rank(px["EEM"], 63) <= 5) & (rank(px["FXI"], 21) >= 90)
for t in ["EEM", "FXI", "EFA"]:
    score("EEM 63d rank<=5 & FXI 21d rank>=90", em_div, t)

print("=" * 100)
print("STATE 6  energy one-day thrust (USO 1d >= +5%)")
print("=" * 100)
uso_pop = px["USO"].pct_change() >= 0.05
for t in ["USO", "XLE", "DBC"]:
    score("USO 1d >= +5%", uso_pop, t)

print("=" * 100)
print("STATE 7  semis deep 63d laggard (SMH 63d rank <= 2)  [registry: snapback dead]")
print("=" * 100)
smh_lag = rank(px["SMH"], 63) <= 2
for t in ["SMH", "XLK"]:
    score("SMH 63d rank <= 2", smh_lag, t)

print("=" * 100)
print("STATE 8  dollar washout inside an uptrend (DX z10 <= -1.5, 63d rank >= 60)")
print("=" * 100)
dxs = px["DX-Y.NYB"].dropna()
z10 = (dxs.pct_change(10)) / (dxs.pct_change().rolling(21).std() * np.sqrt(10))
dx_wash = (z10 <= -1.5) & (rank(dxs, 63) >= 60)
score("DX z10<=-1.5 & 63d rank>=60", dx_wash, "DX-Y.NYB", "[2026-08-10 killed a near-identical cell]")
