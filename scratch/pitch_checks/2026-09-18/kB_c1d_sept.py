"""c1 round 2, last attack - does the gate survive September's own drag?

The live anchor is a SEPTEMBER quad. The ungated Sep post-quad window is 7-19
at h=8 (T3's short edge) and the gated cell holds ONE September (2001-09-21,
the post-9/11 reopening). Estimate the September expectation from the QUAD
cell itself (month-demeaned), list the quad-month rows, and report the
ex-September cell that would carry a park.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["IWM", "SPY"])
cal = px["SPY"].dropna().index
px = px.reindex(cal)


def z_sleeve(s, n=10):
    c = s.dropna()
    vol21 = c.pct_change().rolling(21).std()
    return (c.pct_change(n) / (vol21 * np.sqrt(n))).reindex(s.index)


Z = z_sleeve(px["IWM"])


def expiry_session(d):
    loc = int(cal.searchsorted(d))
    return loc if (loc < len(cal) and cal[loc] == d) else loc - 1


QUADS = sorted({expiry_session(d) for d in load_events(["quad_witching"])["date"]
                if cal[0] <= d <= cal[-1]})

for h in (3, 5, 8, 10):
    df = pd.DataFrame({"d": [cal[q] for q in QUADS],
                       "m": [cal[q].month for q in QUADS],
                       "z": [Z.iloc[q - 1] for q in QUADS],
                       "y": [np.nan if q + h >= len(cal) else px["IWM"].iloc[q + h] / px["IWM"].iloc[q] - 1
                             for q in QUADS]}).dropna()
    mm = df.groupby("m")["y"].mean()
    cm = df[df.z > -1].groupby("m")["y"].mean()
    df["dm"] = df["y"] - df["m"].map(mm)
    df["dmc"] = df["y"] - df["m"].map(cm)
    g = df[df.z <= -1]
    w = int((g.dm > 0).sum())
    print(f"\nh={h}: quad month ungated means {dict((k, round(100*v, 2)) for k, v in mm.items())}")
    print(f"   gate effect vs own-month ungated mean {100*g.dm.mean():+.3f}pp ({w}-{len(g)-w}, "
          f"sign p {sign_test(w, len(g)):.4f}); vs own-month COMPLEMENT {100*g.dmc.mean():+.3f}pp")
    print(f"   Sept-adjusted expectation: vs ungated {100*(mm[9]+g.dm.mean()):+.3f}% | "
          f"vs complement {100*(cm[9]+g.dmc.mean()):+.3f}%")
    for m in (3, 6, 9, 12):
        gm = g[g.m == m]
        print(f"   month {m:2d}: gated {len(gm)} {[round(100*x, 2) for x in gm.y]}  "
              f"compl mean {100*cm[m]:+.2f}% ({int((df[(df.m==m)&(df.z>-1)].y>0).sum())}-"
              f"{int((df[(df.m==m)&(df.z>-1)].y<=0).sum())})")
    ex = g[g.m != 9]
    w2 = int((ex.y > 0).sum())
    print(f"   EX-SEPTEMBER gated cell: n={len(ex)} mean {100*ex.y.mean():+.3f}% "
          f"rec {w2}-{len(ex)-w2} sign p {sign_test(w2, len(ex)):.4f}; "
          f"ex-Sep complement {100*df[(df.m!=9)&(df.z>-1)].y.mean():+.3f}%")
