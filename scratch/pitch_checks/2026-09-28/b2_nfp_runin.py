"""C2 round 1: NFP run-in, entry at the k=-4 close (signal k=-5), exit at the NFP
close (h=4) or NFP+1 (h=5). Pre-specified legs: SHORT DX (DX-Y.NYB / UUP) and
LONG GLD, gated on DX 21d rank >= 85 and ^TNX within 1% of its 252 high, both
read on the SIGNAL close (k=-5; live = 2026-09-25). Ten-class table ungated and
gated, gate legs separately, tdom-matched non-NFP month-turn control.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

CLS = ["SPY", "IWM", "TLT", "HYG", "GLD", "SLV", "USO", "UUP", "DX-Y.NYB", "EFA", "SVXY", "^VIX"]
raw = load_prices(CLS + ["^TNX"])
IDX = raw["SPY"]["Close"].index
PX = pd.DataFrame({t: raw[t]["Close"].reindex(IDX).ffill(limit=2) for t in CLS})
# SVXY -0.5x era only
PX.loc[PX.index < "2018-03-01", "SVXY"] = np.nan

dx = raw["DX-Y.NYB"]["Close"].dropna()
DXR = pct_rank(dx, 21, 252).reindex(IDX).ffill(limit=2).to_numpy()
tnx = raw["^TNX"]["Close"].dropna()
TNR = (tnx / tnx.rolling(252).max()).reindex(IDX).ffill(limit=2).to_numpy()
live = IDX[-1]
print(f"LIVE {live.date()}: DX r21 {DXR[-1]:.1f} (need >=85), TNX/252hi {TNR[-1]:.4f} (need >=0.99)")

nfp = load_events(["nfp"])["date"]
pos, kept = anchor_positions(IDX, nfp[nfp <= live], 0)
pos = np.array(pos)
pos = pos[(pos - 5 >= 0) & (pos + 1 < len(IDX))]
print(f"NFP anchors {len(pos)}: {IDX[pos[0]].date()}..{IDX[pos[-1]].date()}")
tdom = pd.Series(1, index=IDX).groupby([IDX.year, IDX.month]).cumsum().to_numpy()
print("NFP tdom distribution:", pd.Series(tdom[pos]).value_counts().sort_index().to_dict(),
      " live NFP 2026-10-02 = tdom 2")

G_DX = DXR[pos - 5] >= 85
G_TN = TNR[pos - 5] >= 0.99
G_J = G_DX & G_TN
print(f"gate counts: DX {G_DX.sum()}, TNX {G_TN.sum()}, joint {G_J.sum()}")


def win(t, ends, hx):
    c = PX[t].to_numpy()
    e = ends + hx          # hx=0 -> NFP close, hx=1 -> NFP+1
    return c[e] / c[ends - 4] - 1.0


def cell(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "   n/a"
    w = int((v > 0).sum())
    return f"{100*v.mean():+.3f} {w}-{len(v)-w}"


# tdom-matched non-NFP control: windows ending on the same tdom, no NFP in (entry, exit]
nfp_set = set(pos.tolist())
allend = np.arange(5, len(IDX) - 1)
isn = np.zeros(len(IDX) + 2, bool)
isn[pos] = True
has_nfp = np.array([isn[p - 3:p + 2].any() for p in range(len(IDX))])  # NFP in (p-4, p+1]
ctrl_end = {d: np.array([p for p in allend if tdom[p] == d and not has_nfp[p]])
            for d in range(1, 24)}

rows = []
for t in CLS:
    r = {"class": t}
    for hx, lab in ((0, "h4"), (1, "h5")):
        v = win(t, pos, hx)
        r[f"all_{lab}"] = cell(v)
        r[f"gJ_{lab}"] = cell(v[G_J])
        c = PX[t].to_numpy()
        n = 4 + hx
        dr = c[n:] / c[:-n] - 1
        r[f"drift_{lab}"] = f"{100*np.nanmean(dr):+.3f}"
        # tdom-weighted non-NFP control
        cm = []
        for p in pos:
            ce = ctrl_end[tdom[p]]
            ce = ce[ce + hx < len(IDX)]
            cm.append(np.nanmean(win(t, ce, hx)))
        r[f"tdomC_{lab}"] = f"{100*np.nanmean(cm):+.3f}"
    rows.append(r)
print("\n=== TEN-CLASS TABLE (LONG sign, % mean + record). all=ungated NFP, gJ=joint gate ===")
print(pd.DataFrame(rows).to_string(index=False))

# gate legs separately on the pre-specified vehicles
print("\n=== GATE LEGS (pre-specified sign) ===")
for t, s in (("DX-Y.NYB", -1), ("UUP", -1), ("GLD", 1)):
    out = []
    for hx in (0, 1):
        v = s * win(t, pos, hx)
        for lab, m in (("ungated", np.ones(len(pos), bool)), ("DX leg", G_DX), ("TNX leg", G_TN),
                       ("joint", G_J), ("DX & not TNX", G_DX & ~G_TN), ("not DX", ~G_DX)):
            vv = v[m]
            vv2 = vv[~np.isnan(vv)]
            w = int((vv2 > 0).sum())
            d = summarize(vv, f"{'h4' if hx == 0 else 'h5'} {lab}")
            d["rec"] = f"{w}-{len(vv2)-w}"
            d["sign_p"] = sign_test(w, len(vv2))
            out.append(d)
    show(out, f"{t} sign {s:+d}")

# same gate on ALL days (is it the NFP or the gate?) and on the non-NFP tdom-matched windows
print("\n=== GATE ON ANY DAY / ON NON-NFP MONTH TURNS (pre-specified sign, day-level) ===")
for t, s in (("DX-Y.NYB", -1), ("UUP", -1), ("GLD", 1)):
    c = PX[t].to_numpy()
    out = []
    for hx in (0, 1):
        n = 4 + hx
        sig = np.arange(0, len(IDX) - n - 1)
        fr = s * (c[sig + 1 + n] / c[sig + 1] - 1)
        gj = (DXR[sig] >= 85) & (TNR[sig] >= 0.99)
        gd = DXR[sig] >= 85
        out.append(summarize(fr, f"h{n} all days"))
        out.append(summarize(fr[gd], f"h{n} any day, DX gate"))
        out.append(summarize(fr[gj], f"h{n} any day, joint gate"))
        ce = np.concatenate([ctrl_end[d] for d in range(1, 9)])
        ce = ce[ce + hx < len(IDX)]
        vc = s * win(t, ce, hx)
        gce = (DXR[ce - 5] >= 85) & (TNR[ce - 5] >= 0.99)
        out.append(summarize(vc, f"h{n} non-NFP tdom1-8 turn"))
        out.append(summarize(vc[gce], f"h{n} non-NFP turn, joint gate"))
        out.append(summarize(vc[DXR[ce - 5] >= 85], f"h{n} non-NFP turn, DX gate"))
    show(out, f"{t} sign {s:+d}")

# era / midterm / October on the pre-specified legs
D = IDX[pos]
mid = np.array([d.year % 4 == 2 for d in D])
octo = np.array([d.month == 10 for d in D])
for t, s in (("DX-Y.NYB", -1), ("GLD", 1)):
    out = []
    for hx in (0, 1):
        v = s * win(t, pos, hx)
        L = "h4" if hx == 0 else "h5"
        for lab, m in (("DX gate", G_DX), ("joint", G_J)):
            out += [summarize(v[m & (D < "2018-01-01")], f"{L} {lab} pre-2018"),
                    summarize(v[m & (D >= "2018-01-01")], f"{L} {lab} 2018+"),
                    summarize(v[m & mid], f"{L} {lab} midterm"),
                    summarize(v[m & octo], f"{L} {lab} October prints")]
        out += [summarize(v[mid], f"{L} ungated midterm"), summarize(v[octo], f"{L} ungated October")]
    show(out, f"splits {t} sign {s:+d}")

print("\n=== JOINT-GATE EPISODES (short DX, long GLD, h4/h5) ===")
for p in pos[G_J]:
    print(f"  NFP {IDX[p].date()}  DXr21 {DXR[p-5]:.1f} TNX/hi {TNR[p-5]:.4f}  "
          f"-DX h4 {-100*win('DX-Y.NYB', np.array([p]), 0)[0]:+.3f} h5 {-100*win('DX-Y.NYB', np.array([p]), 1)[0]:+.3f}  "
          f"GLD h4 {100*win('GLD', np.array([p]), 0)[0]:+.3f} h5 {100*win('GLD', np.array([p]), 1)[0]:+.3f}")
print("\n=== DX-GATE EPISODES ===")
for p in pos[G_DX]:
    print(f"  NFP {IDX[p].date()}  DXr21 {DXR[p-5]:.1f} TNX/hi {TNR[p-5]:.4f}  "
          f"-DX h4 {-100*win('DX-Y.NYB', np.array([p]), 0)[0]:+.3f} h5 {-100*win('DX-Y.NYB', np.array([p]), 1)[0]:+.3f}  "
          f"GLD h5 {100*win('GLD', np.array([p]), 1)[0]:+.3f}")
