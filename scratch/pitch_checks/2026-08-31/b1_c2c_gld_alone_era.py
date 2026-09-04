"""C2 round 3 -- close the loop on the surface round 2 exposed.

Round 2 showed the three-name complex clause is worth -0.071pp (h=3) over a
bare "GLD <= -2%" day, i.e. C2's whole-complex premise is empty and what is
left is a one-name gold/dollar inverse. Before writing C2 off, check whether
that RESIDUAL one-name cell is itself alive in the live era, and whether it is
special at all against "any liquid asset down 2% -> long DX".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

NAMES = ["GLD", "SLV", "GDX", "TLT", "USO", "XLE", "SPY", "EEM", "FXI", "XME"]


def mkpanel(t, start=None):
    p = close_panel(t).dropna()
    return p.loc[start:] if start else p


def dret(s):
    return s / s.shift(1) - 1.0


def cell(ret, px, mask, label, gap=5):
    s = px.index[mask.reindex(px.index, fill_value=False).values & ret.notna().values]
    if not len(s):
        return {"label": label, "n": 0}
    e = declusters(s, gap, px.index)
    r = summarize(ret.loc[e].values, label)
    r["n_days"] = len(s)
    w = int((ret.loc[e].values > 0).sum())
    r["sign_p"] = round(sign_test(w, len(e)), 4)
    return r


def main():
    px = mkpanel(NAMES + ["DX-Y.NYB"], "2006-05-22")
    r = {t: dret(px[t]) for t in px.columns}
    yr = pd.Series(px.index.year, index=px.index)
    legs = [("DX-Y.NYB", 1.0)]
    dxu = r["DX-Y.NYB"] > 0

    print("########## E1  is the residual GLD-alone cell alive 2018+? ##########")
    g = r["GLD"] <= -0.02
    for h in (3, 5):
        ret = vehicle_ret(px, legs, h, lag=1)
        show([cell(ret, px, g, "GLD <= -2%, all"),
              cell(ret, px, g & (yr < 2018), "GLD <= -2%, pre-2018"),
              cell(ret, px, g & (yr >= 2018), "GLD <= -2%, 2018+"),
              cell(ret, px, g & (yr >= 2018) & dxu, "GLD <= -2%, 2018+ AND DX up"),
              cell(ret, px, g & ~yr.between(2008, 2011), "GLD <= -2%, drop 2008-2011")],
             f"E1 LONG DX after a bare GLD -2% day, h={h}")

    print("\n########## E2  is GOLD special? same rule on 10 liquid names ##########")
    for h in (3, 5):
        ret = vehicle_ret(px, legs, h, lag=1)
        base = ret.dropna()
        rows = []
        for t in NAMES:
            c = cell(ret, px, r[t] <= -0.02, f"{t} <= -2%")
            if c.get("n"):
                c["excess_pct"] = round(c["mean_pct"] - 100 * base.mean(), 3)
            rows.append(c)
        show(rows, f"E2 LONG DX after ANY single name drops 2%, h={h}")


if __name__ == "__main__":
    main()
