"""D8b - is the opex-anchored XLF/SPY result an XLF effect or just "SPY rallies
into opex"?

d8_xlf_opex.py section 7 found the REVERSE of the candidate (short XLF / long
SPY, exit at the next opex close) at -1.03%/episode, t -2.34, N=11. But the leg
attribution showed XLF ~0.00% and SPY +1.03%, i.e. the whole thing may be an
index-into-opex effect with the XLF leg contributing nothing. This script runs
the missing control: the SAME opex-anchored window with NO XLF trigger, and the
month composition of the 11 episodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import bootstrap_p_le0, close_panel, declusters, era_split, load_events, pct_rank, show, summarize  # noqa: E402

px = close_panel(["XLF", "SPY"]).dropna()
cal = px.index
OPEX = pd.DatetimeIndex(load_events(["opex"])["date"])
opos = np.array([cal.get_loc(d) for d in OPEX if d in cal])


def anchored(mask: np.ndarray | None):
    """Entry at close D+1 where D+1 is 8-12 td before an opex close."""
    dts, xl, sp, mo = [], [], [], []
    for j in opos:
        for k in range(8, 13):
            i1 = j - k
            if i1 < 1:
                continue
            d = cal[i1 - 1]  # signal close
            if mask is not None and not mask[i1 - 1]:
                continue
            dts.append(d); mo.append(cal[j].month)
            xl.append(px["XLF"].iloc[j] / px["XLF"].iloc[i1] - 1.0)
            sp.append(px["SPY"].iloc[j] / px["SPY"].iloc[i1] - 1.0)
    return pd.DatetimeIndex(dts), np.array(xl), np.array(sp), np.array(mo)


trg = ((pct_rank(px["XLF"], 63) >= 95) &
       ((1 - px["XLF"] / px["XLF"].rolling(252).max()) <= 0.01)).reindex(cal).fillna(False).values

dC, xC, sC, mC = anchored(None)          # control: every opex window, no trigger
dT, xT, sT, mT = anchored(trg)           # the candidate's cell

for lab, d, x, s in (("CONTROL all opex windows", dC, xC, sC), ("XLF-triggered", dT, xT, sT)):
    e = np.isin(d.values, declusters(d, 10, cal).values)
    show([summarize((x - s)[e], f"{lab}: pair XLF-SPY"),
          summarize(x[e], f"{lab}:   leg XLF"),
          summarize(s[e], f"{lab}:   leg SPY")],
         f"episodes, {lab} (N_day={len(d)}, N_ep={int(e.sum())})")
    print(f"   pair bootstrap P(mean<=0) = {bootstrap_p_le0((x - s)[e]):.3f}")

# ---- month composition of the triggered episodes -------------------------
e = np.isin(dT.values, declusters(dT, 10, cal).values)
tab = pd.DataFrame({"signal": dT[e], "opex_month": mT[e], "pair_pct": 100 * (xT - sT)[e],
                    "xlf_pct": 100 * xT[e], "spy_pct": 100 * sT[e]})
print("\n=== the 11 triggered episodes, one row each ===")
print(tab.round(2).to_string(index=False))
print("\nby opex month:")
print(tab.groupby("opex_month")["pair_pct"].agg(["count", "mean"]).round(2).to_string())

# ---- is SPY-into-opex itself the whole story? ----------------------------
eC = np.isin(dC.values, declusters(dC, 10, cal).values)
allday = (px["SPY"].shift(-10) / px["SPY"] - 1.0).dropna().values
show([summarize(sC[eC], "SPY, opex-anchored windows"), summarize(allday, "SPY, all-days h=10"),
      summarize(sT[e], "SPY, opex+XLF-trigger")],
     "is the SPY leg doing anything special into opex?")

# ---- and the XLF leg, is it special? -------------------------------------
xall = (px["XLF"].shift(-10) / px["XLF"] - 1.0).dropna().values
show([summarize(xC[eC], "XLF, opex-anchored"), summarize(xall, "XLF, all-days h=10"),
      summarize(xT[e], "XLF, opex+trigger")], "and the XLF leg")

# era for the triggered pair
show(era_split(dT[e], (xT - sT)[e]), "era split, triggered opex-anchored pair")
