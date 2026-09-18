"""c1 round 2 continued - is TODAY in the cell, and what does September do to it?

kB_c1b_gate.py showed the washout-into-opex cell holds on every definition, but
the live reading is marginal (sleeve z10 -1.097 vs -1.0; r10 rank 12.7 vs <=10;
z5 -0.31) and the all-opex (-1,-0.75] bin paid less than the complement.

  G. dose ladders (bins) on each definition, all opex and quads, h=5/8/10
  H. live-neighbourhood cells: episodes whose statistic sits within a band of
     today's value (sleeve z10 +/-0.25, pl z +/-0.25, r10 rank +/-5, z5 +/-0.5)
  I. month-demeaned gate effect -> September-adjusted expectation
  J. September second half generic: IWM z10 <= -1 on ANY day Sep 10..Sep 25,
     lag-1, h=8 (does a September washout bounce anywhere?)
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
pos = pd.Series(range(len(cal)), index=cal)


def z_sleeve(s, n=10):
    c = s.dropna()
    vol21 = c.pct_change().rolling(21).std()
    return (c.pct_change(n) / (vol21 * np.sqrt(n))).reindex(s.index)


DEF = {"z10": z_sleeve(px["IWM"]), "plz": zscore(px["IWM"], 10),
       "r10": pct_rank(px["IWM"], 10), "z5": z_sleeve(px["IWM"], 5)}
LIVE = {k: v.loc["2026-09-17"] for k, v in DEF.items()}
BAND = {"z10": 0.25, "plz": 0.25, "r10": 5.0, "z5": 0.5}
print("LIVE:", {k: round(v, 3) for k, v in LIVE.items()})


def expiry_session(d):
    loc = int(cal.searchsorted(d))
    return loc if (loc < len(cal) and cal[loc] == d) else loc - 1


def anchors(kind):
    ev = load_events([kind])["date"]
    return sorted({expiry_session(d) for d in ev if cal[0] <= d <= cal[-1]})


QUADS, OPEX = anchors("quad_witching"), anchors("opex")


def r_q(q, h):
    return np.nan if q + h >= len(cal) else px["IWM"].iloc[q + h] / px["IWM"].iloc[q] - 1


def st(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
    return r


# G. dose ladders
BINS = {"z10": [-9, -1.5, -1.25, -1.0, -0.75, -0.5, 0, 9],
        "plz": [-9, -1.5, -1.25, -1.0, -0.75, -0.5, 0, 9],
        "r10": [0, 5, 10, 15, 20, 30, 50, 101],
        "z5": [-9, -1.5, -1.0, -0.5, 0, 9]}
for fam_name, fam in (("ALL OPEX", OPEX), ("QUADS", QUADS)):
    for dname, s in DEF.items():
        rows = []
        x = np.array([s.iloc[q - 1] for q in fam])
        for h in (5, 8, 10):
            y = np.array([r_q(q, h) for q in fam])
            b = BINS[dname]
            for lo, hi in zip(b[:-1], b[1:]):
                m = (x > lo) & (x <= hi)
                if dname == "r10":
                    m = (x > lo) & (x <= hi)
                r = st(y[m], f"h={h} {dname} ({lo},{hi}]")
                rows.append(r)
        show(rows, f"G. dose ladder {fam_name} on {dname} (live {LIVE[dname]:.2f})")

# H. live neighbourhood
rows = []
for fam_name, fam in (("ALL OPEX", OPEX), ("QUADS", QUADS)):
    for dname, s in DEF.items():
        x = np.array([s.iloc[q - 1] for q in fam])
        m = np.abs(x - LIVE[dname]) <= BAND[dname]
        for h in (5, 8, 10):
            y = np.array([r_q(q, h) for q in fam])
            yall = y[~np.isnan(y)]
            r = st(y[m], f"{fam_name} {dname} live+/-{BAND[dname]} h={h}")
            r["all_anchor_pct"] = round(100 * yall.mean(), 3)
            rows.append(r)
show(rows, "H. live-neighbourhood cells")
for fam_name, fam in (("ALL OPEX", OPEX),):
    x = np.array([DEF["z10"].iloc[q - 1] for q in fam])
    m = np.abs(x - LIVE["z10"]) <= 0.25
    print("  z10 live-band episodes (opex):",
          ", ".join(f"{cal[q].date()}({x[i]:+.2f},{100*r_q(q,8):+.2f})" for i, q in enumerate(fam) if m[i]))

# I. month-demeaned gate effect
print("\nI. month-demeaned gate effect (all opex; each gated episode minus its calendar-month ungated mean)")
for h in (5, 8, 10):
    df = pd.DataFrame({"m": [cal[q].month for q in OPEX],
                       "z": [DEF["z10"].iloc[q - 1] for q in OPEX],
                       "y": [r_q(q, h) for q in OPEX]}).dropna()
    mm = df.groupby("m")["y"].mean()
    df["dm"] = df["y"] - df["m"].map(mm)
    g = df[df.z <= -1]
    sep_base = mm.loc[9]
    eff = g["dm"].mean()
    w = int((g["dm"] > 0).sum())
    print(f"  h={h}: gate effect (month-demeaned) {100*eff:+.3f}pp over {len(g)} "
          f"({w}-{len(g)-w}, sign p {sign_test(w, len(g)):.4f}); Sept ungated {100*sep_base:+.3f}% "
          f"-> Sept-adjusted expectation {100*(sep_base+eff):+.3f}%")
    # quads only
    gq = g[[True if 1 else 0 for _ in range(len(g))]]

# J. September second-half washouts on any day
print("\nJ. IWM z10 <= -1 on ANY day, signal Sep 10..Sep 25, lag-1 entry, h=8 (declustered 8)")
z = DEF["z10"]
for h in (5, 8):
    ret = fwd_lag(px["IWM"], h, 1)
    msk = (z <= -1) & (cal.month == 9) & (cal.day >= 10) & (cal.day <= 25) & ret.notna()
    days = cal[msk.values]
    ep = declusters(days, h, cal)
    v = ret.loc[ep].values
    r = st(v, f"Sep 10-25 washout h={h}")
    base = ret[(cal.month == 9) & (cal.day >= 10) & (cal.day <= 25)].dropna()
    print(f"  {r}  | all Sep10-25 days h={h} {100*base.mean():+.3f}%")
    print("   eps:", ", ".join(f"{d.date()}({100*ret[d]:+.2f})" for d in ep))
    # same window other months (day 10..25) for contrast
    msk2 = (z <= -1) & (cal.month != 9) & (cal.day >= 10) & (cal.day <= 25) & ret.notna()
    ep2 = declusters(cal[msk2.values], h, cal)
    print(f"   other months day 10-25 washout: {st(ret.loc[ep2].values, '')}")
