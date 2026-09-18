"""c9 KILL CHECK round 1 - short crude from mid-September into the October
refinery-turnaround trough, conditioned on a 21d thrust (USO 21d rank >= 80).

Mechanism: fall refinery maintenance (mid-Sep through Oct) cuts crude runs, so
crude inventories build and prompt crude weakens. It must show as a MONTH
LADDER with September (and October) distinct, in the modern era, own drift
removed. Then the thrust conditioner with gate attribution.

Entry position matched to today: signal D = tdom 12, entry MOC at the tdom-13
close (2026-09-18 is September's 13th session), exit h = 5, 8, 10 sessions.
Short return = -(vehicle return). USO is the honest vehicle (it holds and rolls
futures); CL=F is the continuous front month, whose roll-day jumps are NOT a
holder's P&L (backwardation makes the continuous series fall at each roll) -
the CL=F minus USO gap on the same windows is reported.
Placebo: offset ladder k=-5..+5 around Sep tdom 13.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["USO", "CL=F", "XLE", "DBC", "SPY"]
P = load_prices(TK)
cal = P["SPY"].index
C = pd.DataFrame({t: P[t]["Close"] for t in TK}).reindex(cal)
print("CL=F NaN on NYSE sessions:", int(C["CL=F"].isna().sum()), "(ffill limit 1)")
C["CL=F"] = C["CL=F"].ffill(limit=1)
pos = pd.Series(range(len(cal)), index=cal)
ym = pd.Series([(d.year, d.month) for d in cal], index=cal)
tdom = ym.groupby(ym).cumcount() + 1  # 1-based session of month

RANK21 = {t: pct_rank(C[t], 21) for t in ("USO", "CL=F")}
print(f"LIVE 2026-09-17: tdom {tdom.loc['2026-09-17']}, USO r21 {RANK21['USO'].loc['2026-09-17']:.1f}, "
      f"CL=F r21 {RANK21['CL=F'].loc['2026-09-17']:.1f}")


def st(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    r = summarize(v, label)
    if r["n"]:
        w = int((v > 0).sum())
        r["rec"] = f"{w}-{len(v)-w}"
        r["sign_p"] = round(sign_test(w, len(v)), 4)
        srt = np.sort(v)[::-1]
        tot = v.sum()
        r["top2_sh"] = round(100 * srt[:2].sum() / tot, 0) if tot > 0 else np.nan
    return r


def entries(month, td=13, shift=0, years=None):
    """cal positions of the tdom-`td` session (+shift) in `month` for each year."""
    out = []
    for (y, m), g in ym.groupby(ym):
        if m != month or (years and y not in years):
            continue
        idx = g.index
        if len(idx) < td:
            continue
        p = pos[idx[td - 1]] + shift
        out.append(p)
    return out


def short_ret(t, p, h):
    if p + h >= len(cal) or p < 0:
        return np.nan
    a, b = C[t].iloc[p], C[t].iloc[p + h]
    return -(b / a - 1) if (a == a and b == b) else np.nan


drift = {(t, h): (C[t].shift(-h) / C[t] - 1).dropna().mean() for t in TK for h in (5, 8, 10)}

# ------------------------------------------------------------ 1. month ladder
for t, y0 in (("USO", 2006), ("CL=F", 2001), ("XLE", 2001), ("DBC", 2006)):
    rows = []
    for h in (5, 10):
        for m in range(1, 13):
            ps = [p for p in entries(m) if cal[p].year >= y0 and cal[p].year <= 2025]
            v = np.array([short_ret(t, p, h) for p in ps])
            r = st(v, f"{t} short m{m:02d} h={h}")
            r["excess_pp"] = round(r["mean_pct"] + 100 * drift[(t, h)], 3)  # short minus (-drift)
            rows.append(r)
    show(rows, f"1. month ladder, SHORT {t} from the tdom-13 close ({y0}-2025); excess = short mean + own long drift")

# rank of September among the 12 months, USO and CL=F, by era
print("\n1b. September's rank among 12 months (SHORT excess, 1 = best short), by era")
for t, eras in (("USO", ((2006, 2014), (2015, 2025), (2006, 2025))),
                ("CL=F", ((2001, 2014), (2015, 2025), (2001, 2025)))):
    for lo, hi in eras:
        for h in (5, 8, 10):
            ex = {}
            for m in range(1, 13):
                ps = [p for p in entries(m) if lo <= cal[p].year <= hi]
                ex[m] = np.nanmean([short_ret(t, p, h) for p in ps])
            srt = sorted(ex, key=lambda k: -ex[k])
            print(f"  {t} {lo}-{hi} h={h}: Sep {100*ex[9]:+.2f}% rank {srt.index(9)+1}, "
                  f"Oct {100*ex[10]:+.2f}% rank {srt.index(10)+1}; best month {srt[0]} {100*ex[srt[0]]:+.2f}%")

# ------------------------------------------------------------ 2. September cell per year + roll gap
print("\n2. September tdom-13 entry, per year: SHORT returns h=5/10 on USO and CL=F, CL-USO gap, USO r21 at D")
rows = []
for p in entries(9):
    y = cal[p].year
    if y > 2025:
        continue
    rows.append({"year": y, "entry": cal[p].date(), "r21": round(RANK21["USO"].iloc[p - 1], 1)
                 if y >= 2007 else np.nan,
                 "USO_h5": round(100 * short_ret("USO", p, 5), 2), "USO_h10": round(100 * short_ret("USO", p, 10), 2),
                 "CL_h10": round(100 * short_ret("CL=F", p, 10), 2),
                 "CLminusUSO_h10": round(100 * (short_ret("CL=F", p, 10) - short_ret("USO", p, 10)), 2)
                 if y >= 2006 else np.nan,
                 "XLE_h10": round(100 * short_ret("XLE", p, 10), 2)})
S = pd.DataFrame(rows).set_index("year")
print(S.to_string())

# ------------------------------------------------------------ 3. thrust gate attribution
print("\n3. thrust gate (USO 21d rank >= 80 at D) - Sept cell, complement, other months, any day")
rows = []
for h in (5, 8, 10):
    ps = [p for p in entries(9) if 2007 <= cal[p].year <= 2025]
    g = [short_ret("USO", p, h) for p in ps if RANK21["USO"].iloc[p - 1] >= 80]
    c = [short_ret("USO", p, h) for p in ps if RANK21["USO"].iloc[p - 1] < 80]
    rows += [st(g, f"Sep tdom13 thrust h={h}"), st(c, f"Sep tdom13 complement h={h}")]
    om = [short_ret("USO", p, h) for m in range(1, 13) if m != 9 for p in entries(m)
          if 2007 <= cal[p].year <= 2025 and RANK21["USO"].iloc[p - 1] >= 80]
    rows.append(st(om, f"other-month tdom13 thrust h={h}"))
    # any-day thrust (declustered), lag-1
    rr = -fwd_lag(C["USO"], h, 1)
    msk = (RANK21["USO"] >= 80) & rr.notna() & (cal.year >= 2007)
    ep = declusters(cal[msk.values], h, cal)
    rows.append(st(rr.loc[ep].values, f"any-day thrust declustered h={h}"))
    rows.append({"label": f"USO own drift h={h} (long)", "mean_pct": 100 * drift[("USO", h)]})
show(rows)
# CL=F version with longer history
rows = []
for h in (5, 10):
    ps = [p for p in entries(9) if 2002 <= cal[p].year <= 2025]
    g = [short_ret("CL=F", p, h) for p in ps if RANK21["CL=F"].iloc[p - 1] >= 80]
    c = [short_ret("CL=F", p, h) for p in ps if RANK21["CL=F"].iloc[p - 1] < 80]
    rows += [st(g, f"CL=F Sep tdom13 thrust h={h}"), st(c, f"CL=F Sep complement h={h}")]
show(rows)

# ------------------------------------------------------------ 4. offset ladder around Sep tdom 13
print("\n4. offset ladder, SHORT USO (2006-2025) and CL=F (2001-2025), entry at Sep tdom 13 + k")
for t, y0 in (("USO", 2006), ("CL=F", 2001)):
    for h in (5, 10):
        res = []
        for k in range(-5, 6):
            v = [short_ret(t, p, h) for p in entries(9, shift=k) if y0 <= cal[p - k].year <= 2025]
            res.append((k, 100 * np.nanmean(v)))
        m0 = res[5][1]
        rk = 1 + sum(1 for _, m in res if m > m0)
        print(f"  {t} h={h}: " + " ".join(f"k{k:+d}:{m:+.2f}" for k, m in res) + f" -> k=0 rank {rk} of 11")
    # thrust-gated ladder (gate re-evaluated at each shifted D)
    for h in (5, 10):
        res = []
        for k in range(-5, 6):
            v = [short_ret(t, p, h) for p in entries(9, shift=k)
                 if max(y0, 2007) <= cal[p - k].year <= 2025 and RANK21[t].iloc[p - 1] >= 80]
            res.append((k, len(v), 100 * np.nanmean(v) if v else np.nan))
        m0 = res[5][2]
        rk = 1 + sum(1 for _, _, m in res if m > m0)
        print(f"  {t} THRUST h={h}: " + " ".join(f"k{k:+d}:{m:+.2f}({n})" for k, n, m in res)
              + f" -> k=0 rank {rk} of 11")
