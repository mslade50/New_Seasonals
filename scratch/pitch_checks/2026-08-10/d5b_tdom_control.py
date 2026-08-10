"""D5 kill confirmation -- is 'TLT into CPI' just TLT's day-of-month profile?

Section G of d5_cpi_tlt_parent.py found TLT's OWN unconditional h=3 return is
-0.152% on calendar days 1-7 and +0.145% on days 14-20. CPI lands mid-month.
That makes the all-days control (+0.052%) the WRONG control: it averages a
band TLT never trades in during the CPI window with the band it does.

This script builds the right control three ways and reports what is left.
It also explains the placebo table in one line: the anchors that "work" are
the mid-to-late-month ones and the anchor that inverts (NFP) is the first-week
one. If the ranking of macro anchors is the ranking of their calendar
positions, the anchor is not the finding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
H = 3
r3 = np.full(N, np.nan)
r3[:N - H] = c[H:] / c[:-H] - 1.0

ev = load_events()

# trading-day-of-month ordinal (1 = first session of the calendar month)
ym = pd.Series(idx.year * 100 + idx.month, index=idx)
tdom = ym.groupby(ym.values).cumcount().values + 1
n_in_month = ym.map(ym.value_counts()).values


def pos_of(kind, lo=8, hi=14):
    out = []
    for x in ev[ev.event == kind]["date"]:
        p = int(idx.searchsorted(x, "left"))
        if lo <= p < N - hi:
            out.append(p)
    return np.array(sorted(set(out)))


cpi_p = pos_of("cpi")
K = 2                       # today's executable entry: close of print-2
entry = cpi_p - K
win = set()
for p in cpi_p:
    win.update(range(p - K, p - K + H + 1))
in_win = np.array([i in win for i in range(N)])

print("=" * 96)
print("1. TLT's OWN h=3 return by TRADING-day-of-month  (no event anywhere)")
print("=" * 96)
rows = []
for j in range(1, 23):
    m = (tdom == j) & ~np.isnan(r3)
    if m.sum() < 30:
        continue
    v = r3[m]
    rows.append({"tdom": j, "N": int(m.sum()), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "t": round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
prof = pd.DataFrame(rows)
print(prof.to_string(index=False))
print(f"\n  all-days ctrl {100*np.nanmean(r3):+.4f}%")
etd = pd.Series(tdom[entry])
print(f"  CPI ENTRY (print-{K}) trading-day-of-month: min {etd.min()} "
      f"median {int(etd.median())} max {etd.max()}  "
      f"mode {etd.mode().tolist()}")
print("  -> CPI entries land where TLT's own drift is at its strongest.")

print("\n" + "=" * 96)
print("2. THREE CONTROLS for the k=2 h=3 CPI cell")
print("=" * 96)
cell_v = r3[entry]
cell_v = cell_v[~np.isnan(cell_v)]


def rep(v, lbl):
    w = int((v > 0).sum())
    sd = v.std(ddof=1)
    return {"control": lbl, "N": len(v), "mean_pct": round(100 * v.mean(), 4),
            "hit": round(100 * w / len(v), 1),
            "t": round(v.mean() / (sd / np.sqrt(len(v))), 2)}


rows = [rep(cell_v, "*** THE CELL: CPI entries (print-2) ***"),
        rep(r3[~np.isnan(r3)], "C1 all days, full history")]

# C2: same calendar-day-of-month band, CPI window removed
dom = np.array([d.day for d in idx])
lo_e, hi_e = int(np.percentile(dom[entry], 10)), int(np.percentile(dom[entry], 90))
m = (dom >= lo_e) & (dom <= hi_e) & ~in_win & ~np.isnan(r3)
rows.append(rep(r3[m], f"C2 calendar dom [{lo_e},{hi_e}], CPI window removed"))

# C3: exact trading-day-of-month match, CPI window removed
tset = set(int(x) for x in tdom[entry])
m = np.isin(tdom, list(tset)) & ~in_win & ~np.isnan(r3)
rows.append(rep(r3[m], f"C3 same trading-dom {sorted(tset)}, CPI win removed"))

# C4: per-entry matched -- each CPI entry vs the mean of its own tdom bucket
matched = []
bucket = {}
for j in tset:
    mm = (tdom == j) & ~in_win & ~np.isnan(r3)
    bucket[j] = r3[mm].mean()
for p in entry:
    if not np.isnan(r3[p]):
        matched.append(r3[p] - bucket[int(tdom[p])])
matched = np.array(matched)
rows.append(rep(matched + np.nanmean(r3[entry]) * 0, "C4 per-entry EXCESS over own tdom bucket"))
print(pd.DataFrame(rows).to_string(index=False))

w = int((matched > 0).sum())
print(f"\n  C4 detail: excess over the calendar-matched control = "
      f"{100*matched.mean():+.4f}pp,  record {w}-{len(matched)-w}, "
      f"sign p {sign_test(w, len(matched)):.4f}, "
      f"bootstrap P(mean<=0) {bootstrap_p_le0(matched):.3f}")
print(f"  t on the excess: "
      f"{matched.mean()/(matched.std(ddof=1)/np.sqrt(len(matched))):+.2f}")
print(f"  TLT round trip ~2.5 bps -> excess = {100*100*matched.mean():.1f} bps "
      f"= {100*100*matched.mean()/2.5:.1f}x cost")
mid = np.array([(idx[p].year % 4) == 2 for p in entry if not np.isnan(r3[p])])
print(f"  MIDTERM excess {100*matched[mid].mean():+.4f}pp (N={int(mid.sum())})  |  "
      f"non-midterm {100*matched[~mid].mean():+.4f}pp (N={int((~mid).sum())})")
yr = np.array([idx[p].year for p in entry if not np.isnan(r3[p])])
print(f"  pre-2018 excess {100*matched[yr < 2018].mean():+.4f}pp "
      f"(N={int((yr<2018).sum())})  |  2018+ {100*matched[yr >= 2018].mean():+.4f}pp "
      f"(N={int((yr>=2018).sum())})")
print(f"  ex-2008 excess {100*matched[yr != 2008].mean():+.4f}pp")

print("\n" + "=" * 96)
print("3. THE PLACEBO TABLE IS THE CALENDAR TABLE")
print("   each macro anchor's k=2 h=3 TLT return next to the tdom-matched")
print("   control for the days that anchor happens to sit on")
print("=" * 96)
rows = []
for kind in ["nfp", "cpi", "ppi", "fomc_decision", "opex", "vix_expiry"]:
    p = pos_of(kind)
    e = p - K
    v = r3[e]
    ok = ~np.isnan(v)
    v = v[ok]
    ee = e[ok]
    ts = sorted(set(int(x) for x in tdom[ee]))
    mm = np.isin(tdom, ts) & ~np.isnan(r3)
    ctl = r3[mm].mean()
    rows.append({"anchor": kind, "N": len(v),
                 "entry_dom_med": int(np.median(dom[ee])),
                 "entry_tdom_med": int(np.median(tdom[ee])),
                 "cell_pct": round(100 * v.mean(), 3),
                 "tdom_ctrl_pct": round(100 * ctl, 3),
                 "EXCESS_pp": round(100 * (v.mean() - ctl), 3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  If EXCESS collapses toward zero for every anchor while cell_pct")
print("  tracks the calendar position, the 'event fingerprint' is a month-shape.")

print("\n" + "=" * 96)
print("4. THE PRINT SESSION ITSELF, 2018+  (the PPI checker's era-stable cell)")
print("=" * 96)
d1 = c[1:] / c[:-1] - 1.0
for kind in ["cpi", "ppi"]:
    p = pos_of(kind)
    v = np.array([c[q] / c[q - 1] - 1.0 for q in p])
    y = np.array([idx[q].year for q in p])
    for nm, m in [("all", np.ones(len(v), bool)), ("pre-2018", y < 2018),
                  ("2018+", y >= 2018), ("2023+", y >= 2023)]:
        s = v[m]
        wn = int((s > 0).sum())
        print(f"  {kind} print session {nm:9s} N={len(s):3d} "
              f"{100*s.mean():+.4f}%  hit {100*wn/len(s):5.1f}%  "
              f"sign p {sign_test(wn, len(s)):.4f}  "
              f"(ctrl daily {100*d1.mean():+.4f}%)")
    print()
