"""C9 "Post-NFP into CPI": SPY MOC on the NFP close -> MOC the close before
the next CPI print. Both directions, plus the plain NFP -> +5td horizon.

Controls: (1) SPY unconditional drift at the same horizon over the same window,
(2) an all-macro-event-day baseline at the same horizon.
Conditioners tested SEPARATELY: (a) SPY rank5 >= 90, (b) midterm years.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

px = load_prices(["SPY"])["SPY"]
c = px["Close"].dropna()
cal = c.index
pos = pd.Series(range(len(cal)), index=cal)
ev = load_events()
nfp = [d for d in ev.loc[ev.event == "nfp", "date"] if d in pos.index]
cpi = sorted(ev.loc[ev.event == "cpi", "date"])
allev = sorted({d for d in ev["date"] if d in pos.index})
r5 = pct_rank(c, 5)

print(f"NFP dates that are trading days: {len(nfp)} "
      f"({nfp[0].date()} .. {nfp[-1].date()}); "
      f"dropped {len(ev[ev.event=='nfp']) - len(nfp)} non-trading NFP dates")


def to_cpi_h(d):
    """Trading days from anchor close to the last close BEFORE the next CPI."""
    nxt = next((x for x in cpi if x > d), None)
    if nxt is None:
        return None
    later = cal[cal >= nxt]
    if len(later) == 0:
        return None
    return int(pos[later[0]] - pos[d] - 1)


def ret_h(d, h):
    p = pos[d]
    if h is None or h < 1 or p + h >= len(cal):
        return np.nan
    return c.iloc[p + h] / c.iloc[p] - 1.0


hs = {d: to_cpi_h(d) for d in nfp}
valid = [d for d in nfp if hs[d] and hs[d] >= 1 and pos[d] + hs[d] < len(cal)]
hv = np.array([hs[d] for d in valid])
print(f"horizon-to-next-CPI distribution over {len(valid)} NFPs: "
      f"min={hv.min()} med={int(np.median(hv))} max={hv.max()}  "
      f"(today's = 3)")

cell = np.array([ret_h(d, hs[d]) for d in valid])
dts = pd.DatetimeIndex(valid)

# --- control 1: own unconditional drift, horizon-matched -------------------
ctl_own = []
for h in sorted(set(hv)):
    fr = fwd_ret(c, h).dropna()
    ctl_own.append((h, fr.mean(), len(fr)))
matched = np.mean([dict((h, m) for h, m, _ in ctl_own)[h] for h in hv])

# --- control 2: all macro event days, horizon-matched ----------------------
ev_ctl = []
for d, h in zip(valid, hv):
    pass
allev_rets = {h: np.array([ret_h(d, h) for d in allev]) for h in sorted(set(hv))}
matched_ev = np.mean([np.nanmean(allev_rets[h]) for h in hv])

rows = [summarize(cell, "NFP->pre-CPI (LONG)"),
        summarize(-cell, "NFP->pre-CPI (SHORT)")]
rows.append({"label": "ctl: SPY all-days, h-matched", "n": len(c),
             "mean_pct": 100 * matched})
rows.append({"label": "ctl: all macro events, h-matched",
             "n": len(allev), "mean_pct": 100 * matched_ev})
show(rows, "1) does the cell exist? NFP close -> close before next CPI")

# --- plain +5td and horizon sweep (sign-flip check) ------------------------
sweep = []
for h in [1, 2, 3, 4, 5, 6, 10]:
    v = np.array([ret_h(d, h) for d in nfp])
    b = fwd_ret(c, h).dropna()
    e = np.array([ret_h(d, h) for d in allev])
    s = summarize(v, f"NFP +{h}td")
    s["ctl_alldays_pct"] = 100 * b.mean()
    s["ctl_allevent_pct"] = 100 * np.nanmean(e)
    sweep.append(s)
show(sweep, "4) horizon sweep — sign stability (long side)")

# --- conditioners, reported separately ------------------------------------
def cond_rows(mask_fn, name, hlist=("cpi", 5)):
    out = []
    for h in hlist:
        if h == "cpi":
            sub = [d for d in valid if mask_fn(d)]
            v = np.array([ret_h(d, hs[d]) for d in sub])
            lbl = f"{name} | to-CPI"
        else:
            sub = [d for d in nfp if mask_fn(d) and pos[d] + h < len(cal)]
            v = np.array([ret_h(d, h) for d in sub])
            lbl = f"{name} | +{h}td"
        s = summarize(v, lbl)
        s["p_le0_boot"] = bootstrap_p_le0(v)
        out.append((s, pd.DatetimeIndex(sub), v))
    return out


conds = {
    "uncond": lambda d: True,
    "(a) rank5>=90": lambda d: r5.get(d, np.nan) >= 90,
    "(b) midterm": lambda d: d.year % 4 == 2,
    "(a)+(b)": lambda d: (r5.get(d, np.nan) >= 90) and (d.year % 4 == 2),
}
crows, keep = [], {}
for name, fn in conds.items():
    for s, sd, v in cond_rows(fn, name):
        crows.append(s)
        keep[s["label"]] = (sd, v)
show(crows, "conditioners (separate, so you can see N shrink)")

# --- sensitivity on the rank gate -----------------------------------------
sens = []
for thr in [80, 85, 90, 95]:
    for h in [3, 5]:
        sub = [d for d in nfp if r5.get(d, np.nan) >= thr and pos[d] + h < len(cal)]
        sens.append(summarize(np.array([ret_h(d, h) for d in sub]),
                              f"rank5>={thr} +{h}td"))
show(sens, "4) threshold sensitivity")

# --- declustering + era + CPI-in-window -----------------------------------
print("\n=== 3) declustering ===")
for lbl in ["uncond | to-CPI", "(a) rank5>=90 | to-CPI", "(b) midterm | to-CPI"]:
    sd, v = keep[lbl]
    dc = declusters(sd, 5, cal)
    print(f"  {lbl:28s} day-level N={len(sd)} -> declustered N={len(dc)} "
          f"(min gap between NFPs = "
          f"{int(np.diff([pos[d] for d in sd]).min()) if len(sd)>1 else 'na'} td)")

print("\n=== 2) era stability + worst window ===")
erows = []
for lbl in ["uncond | to-CPI", "uncond | +5td", "(a) rank5>=90 | to-CPI",
            "(a) rank5>=90 | +5td", "(b) midterm | to-CPI", "(b) midterm | +5td"]:
    sd, v = keep[lbl]
    for s in era_split(sd, v):
        s["label"] = f"{lbl} :: {s['label']}"
        erows.append(s)
show(erows, "era split")

print("\n=== 6) CPI inside the window? (+5td variant) ===")
cpiset = set(cpi)
sd, v = keep["uncond | +5td"]
inside = np.array([any(x in cpiset for x in cal[pos[d] + 1: pos[d] + 6]) for d in sd])
show([summarize(v[inside], "+5td, CPI inside"),
      summarize(v[~inside], "+5td, no CPI inside")], "CPI-in-window split")

print("\n=== 2b) ERA-MATCHED control: is the 2018+ NFP cell beating the "
      "2018+ tape? ===")
erow = []
for lo, hi, nm in [("2000-01-01", "2018-01-01", "pre-2018"),
                   ("2018-01-01", "2030-01-01", "2018+")]:
    m = (cal >= lo) & (cal < hi)
    sub = [d for d in valid if lo <= str(d.date()) < hi]
    v = np.array([ret_h(d, hs[d]) for d in sub])
    hsub = np.array([hs[d] for d in sub])
    base = np.mean([fwd_ret(c, h)[m].mean() for h in hsub])
    v5 = np.array([ret_h(d, 5) for d in [d for d in nfp if lo <= str(d.date()) < hi]])
    erow.append({"era": nm, "n": len(sub), "nfp_toCPI_pct": 100 * np.nanmean(v),
                 "alldays_hmatched_pct": 100 * base,
                 "edge_vs_ctl_pct": 100 * (np.nanmean(v) - base),
                 "nfp_+5td_pct": 100 * np.nanmean(v5),
                 "alldays_5td_pct": 100 * fwd_ret(c, 5)[m].mean()})
show(erow, "era-matched vs own-tape control")

print("\n=== 5) cost sanity: SPY ~1bp/side => 2bp round trip = 0.020% ===")
print("     edge must clear ~0.100% (5x). Compare to the means above.")
