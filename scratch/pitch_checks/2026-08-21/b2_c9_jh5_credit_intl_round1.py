"""C9 round 1 — Jackson Hole at JH-5 on CREDIT (HYG, LQD) and INTERNATIONAL
(EFA, EEM, FXI, EWJ). The two classes the JH sweep never reached; the other
five (rates, gold, dollar, small caps, large caps) are closed in the registry.

ANCHOR: JH-5 means the session five trading days BEFORE the symposium's
keynote session. 2026: JH = 2026-08-28, so JH-5 = 2026-08-21 (today).
In lag=1 grammar the signal date is JH-6 and entry is MOC on the JH-5 close.

TWO TRADES, deliberately not blurred:
  h <= 4  exits BEFORE the speech      -> "risk-premium build into the event"
  h >= 5  the exit is ON or AFTER JH   -> "the release itself"
h=5 exits exactly on the JH session close.

MECHANISM to state or refuse: a scheduled Fed-chair speech is a policy-
uncertainty release. The honest prediction is a premium build into it (so the
h<=4 leg should be NEGATIVE for a risk asset, not positive) and a release
after. Credit and EM are the classes most exposed to a dollar-funding
repricing out of it.

Grid: 6 vehicles x 10 horizons = 60 cells, priced at the bottom.
Sample ceiling: one anchor a year, N in the low twenties. sign_test is the
statistic; no t-stat is quoted as evidence.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

CRED = ["HYG", "LQD"]
INTL = ["EFA", "EEM", "FXI", "EWJ"]
VEH = CRED + INTL
COST = {"HYG": 4, "LQD": 4, "EFA": 3, "EEM": 3, "FXI": 4, "EWJ": 4, "SPY": 2}

px = close_panel(VEH + ["SPY"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

jh = load_events(["jackson_hole"])["date"]
JHPOS = []
for d in jh:
    p = int(idx.searchsorted(d))
    if p >= len(idx):          # registry 2026-08-11: never mint a fake anchor
        continue
    JHPOS.append((d.year, p))
print(f"JH sessions resolvable in the price calendar: {len(JHPOS)} "
      f"({JHPOS[0][0]}..{JHPOS[-1][0]})")

K = 5  # JH-5


def anchors(k):
    """signal dates whose lag=1 entry lands on JH-k."""
    out = []
    for y, p in JHPOS:
        j = p - k - 1
        if 0 <= j < len(idx):
            out.append(idx[j])
    return pd.DatetimeIndex(out)


aJH = anchors(K)
print(f"JH-{K} anchors: {len(aJH)}  first {aJH[0].date()} last {aJH[-1].date()}")
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month])
                 .cumcount().values + 1, index=idx)
ATD = sorted(tdom.loc[aJH].unique())
print(f"anchor tdom values: {ATD}")


def R(v, h, lag=1):
    return fwd_lag(px[v].dropna(), h, lag)


print("\n\n===== 1. the grid: JH-5 entry, h=1..10, all 6 vehicles =====")
print("excess = cell mean minus the vehicle's own AUGUST tdom-matched control")
print("(month-of-year control folded in: the control is August only)")
grid = []
for v in VEH + ["SPY"]:
    for h in range(1, 11):
        r = R(v, h).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        if len(a) < 10:
            continue
        ctrl = r.index[(r.index.month.isin([8, 9]))
                       & (tdom.reindex(r.index).isin(ATD))
                       & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
        vals = r.loc[a]
        w = int((vals > 0).sum())
        grid.append({"v": v, "h": h, "n": len(a), "mean": 100 * vals.mean(),
                     "med": 100 * vals.median(),
                     "ctrl": 100 * r.loc[ctrl].mean(),
                     "excess": 100 * (vals.mean() - r.loc[ctrl].mean()),
                     "rec": f"{w}-{len(vals)-w}",
                     "signp": sign_test(w, len(vals)),
                     "signp_dn": sign_test(len(vals) - w, len(vals)),
                     "worst": 100 * vals.min()})
g = pd.DataFrame(grid)
print(g[g.v != "SPY"].round(3).to_string(index=False))
print("\nSPY reference (the class the registry already closed):")
print(g[g.v == "SPY"].round(3).to_string(index=False))

print("\n\n===== 2. the two trades, separated =====")
for lbl, hs in [("PRE-speech (exit before JH, h<=4)", (1, 2, 3, 4)),
                ("STRADDLE (exit on/after JH, h>=5)", (5, 6, 7, 8, 9, 10))]:
    sub = g[(g.v != "SPY") & (g.h.isin(hs))]
    print(f"\n{lbl}: mean excess across the class = "
          f"{sub.excess.mean():+.3f}pp, best cell "
          f"{sub.loc[sub.excess.idxmax(), 'v']} h="
          f"{sub.loc[sub.excess.idxmax(), 'h']} at "
          f"{sub.excess.max():+.3f}pp, worst {sub.excess.min():+.3f}pp")
    print("  per-vehicle mean excess: " + "  ".join(
        f"{v}:{sub[sub.v==v].excess.mean():+.3f}" for v in VEH))
print("\nMECHANISM CHECK: the premium-build story predicts the PRE leg is")
print("NEGATIVE for credit and EM. Read the signs above before believing it.")

print("\n\n===== 3. grid price (60 cells) =====")
cand = g[g.v != "SPY"]
sd = cand.excess.std(ddof=1)
best = cand.loc[cand.excess.abs().idxmax()]
print(f"{len(cand)} cells, excess sd {sd:.3f}pp, "
      f"{int((cand.excess.abs()>1.0).sum())} clear |1.0pp|")
print(f"best |excess| cell: {best.v} h={best.h} at {best.excess:+.3f}pp, "
      f"record {best.rec}, sign p {min(best.signp, best.signp_dn):.4f}, "
      f"worst episode {best.worst:.2f}%")
print(f"  Sidak on the best cell's sign p over {len(cand)} cells: "
      f"{1-(1-min(best.signp, best.signp_dn))**len(cand):.3f}")

print("\n\n===== 4. OFFSET PLACEBO LADDER, entry at JH-k for k=-3..+13 =====")
print("(k=5 is the TRUE anchor; higher k = earlier entry)")
for v in VEH:
    for h in (3, 5, 10):
        lad = {}
        r = R(v, h).dropna()
        for k in range(-3, 14):
            a = anchors(k).intersection(r.index)
            if len(a) < 10:
                continue
            lad[k] = 100 * r.loc[a].mean()
        s = pd.Series(lad)
        if K not in s.index:
            continue
        rk_hi = int((s > s[K]).sum()) + 1
        rk_lo = int((s < s[K]).sum()) + 1
        print(f"{v} h={h:2d}: TRUE k=5 {s[K]:+.3f}% ranks {rk_hi} of {len(s)} "
              f"from the top / {rk_lo} from the bottom | best k={s.idxmax()} "
              f"{s.max():+.3f}%  worst k={s.idxmin()} {s.min():+.3f}%")
        print("      " + " ".join(f"{k}:{val:+.2f}" for k, val in s.items()))

print("\n\n===== 5. midterm split (2026 IS a midterm year) =====")
for v in VEH:
    for h in (3, 5, 10):
        r = R(v, h).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        vals = r.loc[a]
        m = (vals.index.year % 4 == 2)
        mw = int((vals[m] > 0).sum())
        print(f"{v:4s} h={h:2d}: all {100*vals.mean():+.3f}% (N={len(vals)}) | "
              f"midterm {100*vals[m].mean():+.3f}% (N={int(m.sum())}, "
              f"{mw}-{int(m.sum())-mw}) | non-midterm "
              f"{100*vals[~m].mean():+.3f}%")

print("\n\n===== 6. concentration + cost on every cell that is positive at h=5 =====")
for v in VEH:
    for h in (3, 5, 10):
        r = R(v, h).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        vals = r.loc[a].values
        o = np.argsort(-vals)
        row = g[(g.v == v) & (g.h == h)]
        x = float(row.excess.iloc[0]) if len(row) else np.nan
        print(f"{v:4s} h={h:2d}: mean {100*vals.mean():+.3f}% | drop1 "
              f"{100*np.delete(vals,o[:1]).mean():+.3f} drop2 "
              f"{100*np.delete(vals,o[:2]).mean():+.3f} drop3 "
              f"{100*np.delete(vals,o[:3]).mean():+.3f} | excess {x:+.3f}pp = "
              f"{x*100:+.1f}bps = {x*100/COST[v]:+.1f}x a {COST[v]}bp round trip")
