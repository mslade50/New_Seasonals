"""b1 - C4: the ten sessions into Jackson Hole. Legs 4a TLT, 4b GLD, 4c DX.

Anchor mirrors today EXACTLY: today is 2026-08-13 and JH 2026-08-28 sits +11 td
away, so the historical anchor is idx[jh_pos - 11] (every one a Thursday, same
weekday as today). Entry lag=1 (MOC the next close), h=10 -> exit ON the JH
session close. h=9 (exit the session BEFORE JH) reported alongside.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

TK = ["TLT", "IEF", "LQD", "GLD", "GDX", "DX-Y.NYB", "UUP", "SPY"]
px = close_panel(TK)
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

jh = load_events(["jackson_hole"])["date"]
anchors = []
for d in jh:
    p = pos.get(d)
    if p is None:
        later = idx[idx >= d]
        if len(later) == 0:
            continue
        p = pos[later[0]]
    a = p - 11
    if a >= 0:
        anchors.append(idx[a])
anchors = pd.DatetimeIndex(sorted(anchors))
mask = pd.Series(False, index=idx)
mask.loc[anchors] = True
print("JH anchors:", len(anchors), anchors[0].date(), "..", anchors[-1].date())
print("weekdays:", sorted(set(anchors.weekday)))

MID = pd.DatetimeIndex([d for d in anchors if d.year % 4 == 2])
NON = anchors.difference(MID)
print(f"midterm anchors N={len(MID)}: {[d.year for d in MID]}")


def leg_report(name, legs, h=10, cost=2.0):
    print("\n" + "=" * 78)
    battery(px, mask, legs, h, name, cost, min_gap=10,
            event_kinds=("cpi", "ppi", "nfp"))
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    a = pd.DatetimeIndex(anchors).intersection(valid)
    m = pd.DatetimeIndex(MID).intersection(valid)
    n = pd.DatetimeIndex(NON).intersection(valid)
    rows = [summarize(ret.loc[a].values, f"ALL JH (N={len(a)})"),
            summarize(ret.loc[m].values, f"MIDTERM (N={len(m)})"),
            summarize(ret.loc[n].values, f"NON-midterm (N={len(n)})")]
    show(rows, f"MANDATORY cycle-year split :: {name}")
    for lbl, s in (("all", a), ("mid", m), ("non", n)):
        v = ret.loc[s].values
        w = int((v > 0).sum())
        print(f"  {lbl:4s} record {w}-{len(v)-w}  sign p={sign_test(w, len(v)):.4f}"
              f"  mean {100*v.mean():+.3f}%")
    # per-year detail
    per = pd.DataFrame({"year": [d.year for d in a],
                        "anchor": [d.date() for d in a],
                        "ret_pct": (100 * ret.loc[a]).round(2).values,
                        "mid": [d.year % 4 == 2 for d in a]})
    print("\n  per-event:")
    print(per.to_string(index=False))
    # drop-worst / drop-best robustness
    v = ret.loc[a].values
    print(f"  drop best  -> mean {100*np.sort(v)[:-1].mean():+.3f}%")
    print(f"  drop best2 -> mean {100*np.sort(v)[:-2].mean():+.3f}%")
    print(f"  drop worst -> mean {100*np.sort(v)[1:].mean():+.3f}%")
    if len(m):
        vm = ret.loc[m].values
        print(f"  MIDTERM drop best -> {100*np.sort(vm)[:-1].mean():+.3f}% ; "
              f"drop best2 -> {100*np.sort(vm)[:-2].mean():+.3f}%")
    # h=9 (exit session before JH) contrast
    for hh in (5, 9, 10):
        r2 = vehicle_ret(px, legs, hh, 1)
        aa = pd.DatetimeIndex(anchors).intersection(r2.dropna().index)
        mm = pd.DatetimeIndex(MID).intersection(r2.dropna().index)
        print(f"  h={hh:2d}: all N={len(aa)} {100*r2.loc[aa].mean():+.3f}% | "
              f"mid N={len(mm)} {100*r2.loc[mm].mean():+.3f}%")


leg_report("C4a  LONG TLT into JH", [("TLT", 1.0)])
leg_report("C4a' LONG IEF into JH", [("IEF", 1.0)])
leg_report("C4b  LONG GLD into JH", [("GLD", 1.0)])
leg_report("C4c  LONG DX index into JH", [("DX-Y.NYB", 1.0)])
leg_report("C4c- SHORT DX index into JH", [("DX-Y.NYB", -1.0)])
leg_report("C4c'' LONG UUP into JH", [("UUP", 1.0)])
