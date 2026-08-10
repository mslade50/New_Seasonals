"""C2 round 2 -- the cell that SURVIVED gate attribution: long TLT into the PPI
session, UNGATED. h=1, anchor 2td before the print, lag=1 MOC entry the close
before, exit the print session close.

Round 1 said: episodes N=286, +0.115% vs own drift +0.018%, edge +0.098pp,
welch t=+1.96, sign p=0.0105, top-2 episodes = 0% of total, worst -2.60%,
pre-2018 +0.105 / 2018+ +0.133.

This script tries to kill it four ways:
  A. is it PPI or is it ANY macro print (CPI / NFP / FOMC)?  -> novelty
  B. is it the calendar POSITION (mid-month) rather than the event? -> confound
  C. rate regime -- 2002-2020 was one long duration bull.       -> fence
  D. definition neighbours: anchor offset, horizon, IEF, ^TNX.  -> fragility
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "^TNX", "SPY", "LQD"]).dropna(subset=["TLT"])
idx = px.index
ev = load_events(["ppi", "cpi", "nfp", "fomc_decision"])


def anchor(kind: str, offset: int = -2) -> pd.DatetimeIndex:
    out = []
    for x in ev[ev.event == kind]["date"]:
        p = int(idx.searchsorted(x, side="left"))
        if 0 <= p + offset < len(idx):
            out.append(idx[p + offset])
    return pd.DatetimeIndex(sorted(set(out)))


def cell(dates, legs, h, lbl, lag=1, ctrl_dates=None):
    r = vehicle_ret(px, legs, h, lag)
    d = pd.DatetimeIndex(dates).intersection(r.dropna().index)
    v = r.loc[d].values
    if len(v) == 0:
        print(f"{lbl}: EMPTY")
        return None
    s = summarize(v, lbl)
    base = r.dropna()
    ctrl = base.loc[pd.DatetimeIndex(ctrl_dates).intersection(base.index)].values \
        if ctrl_dates is not None else base.values
    w = int((v > 0).sum())
    print(f"{lbl:44s} N={s['n']:4d} mean {s['mean_pct']:+.3f}% "
          f"hit {s['hit']:5.1f}% t {s['t']:+5.2f} worst {s['worst_pct']:+6.2f}% "
          f"| ctrl {100*np.nanmean(ctrl):+.3f}% edge "
          f"{s['mean_pct']-100*np.nanmean(ctrl):+.3f}pp | sign p "
          f"{sign_test(w, len(v)):.4f}")
    return v, d


print("=" * 90)
print("A. NOVELTY: is it PPI, or does duration rally into EVERY macro print?")
print("=" * 90)
for k in ("ppi", "cpi", "nfp", "fomc_decision"):
    cell(anchor(k), [("TLT", 1.0)], 1, f"TLT h=1, {k} session")
print()
for k in ("ppi", "cpi", "nfp"):
    cell(anchor(k), [("IEF", 1.0)], 1, f"IEF h=1, {k} session")
print()
for k in ("ppi", "cpi", "nfp"):
    cell(anchor(k), [("SPY", 1.0)], 1, f"SPY h=1, {k} session (contrast)")

print("\n" + "=" * 90)
print("B. CONFOUND: PPI lands mid-month. Is this a mid-month duration bid?")
print("=" * 90)
r1 = vehicle_ret(px, [("TLT", 1.0)], 1, 1).dropna()
ppi_a = anchor("ppi")
ppi_exit = pd.DatetimeIndex([idx[idx.searchsorted(d) + 2] for d in ppi_a
                             if idx.searchsorted(d) + 2 < len(idx)])
dom = pd.DatetimeIndex(r1.index).day
print("PPI session day-of-month distribution:",
      dict(pd.Series(ppi_exit.day).value_counts().sort_index().head(20)))
lo, hi = int(np.percentile(ppi_exit.day, 5)), int(np.percentile(ppi_exit.day, 95))
print(f"PPI sessions land on days {ppi_exit.day.min()}..{ppi_exit.day.max()} "
      f"(5-95pct {lo}..{hi})")
# control: same calendar window, the ANCHOR two sessions before a day in that band,
# excluding actual PPI anchors
band_exit = r1.index[(dom >= lo) & (dom <= hi)]
band_anchor = pd.DatetimeIndex(
    [idx[idx.searchsorted(d) - 2] for d in band_exit
     if idx.searchsorted(d) - 2 >= 0])
band_only = band_anchor.difference(ppi_a)
cell(ppi_a, [("TLT", 1.0)], 1, f"PPI anchors")
cell(band_only, [("TLT", 1.0)], 1, f"mid-month band {lo}-{hi}, NON-PPI")
cell(r1.index.difference(band_anchor), [("TLT", 1.0)], 1, "outside the band")

print("\n" + "=" * 90)
print("C. RATE REGIME -- 2002-2020 was one duration bull market")
print("=" * 90)
v, d = cell(ppi_a, [("TLT", 1.0)], 1, "PPI, full")
for name, lo_y, hi_y in [("2002-2012", 2002, 2012), ("2013-2020", 2013, 2020),
                         ("2021-2026 (bear/normalisation)", 2021, 2026),
                         ("2022+ only", 2022, 2026)]:
    m = (d.year >= lo_y) & (d.year <= hi_y)
    if m.sum():
        s = summarize(v[m], name)
        w = int((v[m] > 0).sum())
        print(f"  {name:34s} N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
              f"hit {s['hit']:5.1f}% t {s['t']:+5.2f} "
              f"sign p {sign_test(w, int(m.sum())):.4f}")
mid = d.year % 4 == 2
print()
show([summarize(v[mid], f"midterm (N={int(mid.sum())})"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")],
     "midterm split")
print("\nper-year totals (pp):")
yr = pd.Series(100 * v, index=d).groupby(d.year).agg(["sum", "mean", "count"])
print(yr.round(2).to_string())
print("\nyears positive:", int((yr['sum'] > 0).sum()), "/", len(yr))
print(cluster_note(d, v, k=3))
# drop-best-year
for by in yr["sum"].nlargest(2).index:
    m = d.year != by
    print(f"  drop {by}: mean {100*v[m].mean():+.4f}%  (full {100*v.mean():+.4f}%)")

print("\n" + "=" * 90)
print("D. DEFINITION NEIGHBOURS")
print("=" * 90)
for off in (-4, -3, -2, -1):
    cell(anchor("ppi", off), [("TLT", 1.0)], 1,
         f"anchor offset {off} (h=1)")
print()
for h in (1, 2, 3, 5, 10):
    cell(ppi_a, [("TLT", 1.0)], h, f"PPI anchor, h={h}")
print()
cell(ppi_a, [("IEF", 1.0)], 1, "IEF h=1")
cell(ppi_a, [("LQD", 1.0)], 1, "LQD h=1")
cell(ppi_a, [("TLT", 1.0), ("IEF", -1.0)], 1, "TLT - IEF (curve) h=1")
print()
print("lag sensitivity (lag=0 is UNTRADEABLE, shown for contrast):")
for lg in (0, 1, 2):
    cell(ppi_a, [("TLT", 1.0)], 1, f"lag={lg}", lag=lg)

print("\n" + "=" * 90)
print("E. ^TNX check -- does the yield actually FALL on PPI sessions?")
print("=" * 90)
tnx = px["^TNX"].dropna()
chg = tnx.diff()          # in index units; x100 = bps
pe = pd.DatetimeIndex(ppi_exit).intersection(chg.dropna().index)
allc = chg.dropna()
print(f"PPI sessions: mean d^TNX {100*chg.loc[pe].mean():+.2f} bps  "
      f"median {100*chg.loc[pe].median():+.2f} bps  N={len(pe)}  "
      f"down-days {100*(chg.loc[pe] < 0).mean():.1f}%")
print(f"all sessions: mean d^TNX {100*allc.mean():+.2f} bps  "
      f"median {100*allc.median():+.2f} bps  down-days "
      f"{100*(allc < 0).mean():.1f}%")

print("\n" + "=" * 90)
print("F. TAIL / SIZING SANITY")
print("=" * 90)
print(f"mean {100*v.mean():+.3f}%  sd {100*v.std(ddof=1):.3f}%  "
      f"per-trade IR {v.mean()/v.std(ddof=1):.3f}  "
      f"~12 trades/yr -> ann Sharpe {v.mean()/v.std(ddof=1)*np.sqrt(12):.2f}")
print(f"worst 5 episodes: "
      f"{[f'{d[i].date()} {100*v[i]:+.2f}%' for i in np.argsort(v)[:5]]}")
print(f"5th pctile {100*np.percentile(v, 5):+.2f}%   "
      f"1st pctile {100*np.percentile(v, 1):+.2f}%")
net = 100 * v.mean() - 0.03      # 3 bps round trip on TLT
print(f"net of 3bps round trip: {net:+.3f}%  ({net/(100*v.std(ddof=1)/np.sqrt(len(v))):.2f} t)")
