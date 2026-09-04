"""C1 kill attempt 1+2: the registry collision and the Feb-2018 leverage break.

C1 = long SVXY entered MOC on the session before a CPI print, exit MOC +3 td.
Recon claims h=3 +1.473% vs a tdom-matched control of +0.343%.

Two attacks, both about whether the number is a TRADE or an artifact:

  (1) data/pitch_negative_registry.md carries "post-CPI vol crush - the effect
      died after 2018".  Its source is event_seasonality_sweep_2026-08-06.md
      addendum 4, which tested CPI-session OPEN -> +2 close on a SYNTHETIC
      CONSTANT -0.5x leg (scratch/svxy_postevent_grid.py).  Two differences
      from C1: the anchor (after the 08:30 print vs holding through it) and
      the instrument basis.  This script decomposes C1's window into
      [overnight into the print] + [the dead cell] and reports how much of
      C1's edge is the segment already declared dead.

  (2) SVXY was -1x until 2018-02-28 and -0.5x since.  43% of the sample is on
      the doubled basis, so the pooled mean is mechanically inflated.  Rerun
      everything on a constant -0.5x synthetic (halve pre-break daily
      returns), which is the ONLY basis on which today's trade can be sized.

Run: python scratch/pitch_checks/2026-08-11/a1_svxy_cpi_registry_and_leverage.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_lab import (  # noqa: E402
    SVXY_LEV_BREAK, anchor_dates, event_sessions, rebase_half, tdom_control,
    tdom_of,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, declusters, fwd_lag, load_events, load_prices, show,
    sign_test, summarize,
)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

HORIZONS = (1, 2, 3, 5, 10)
OFFSET = 2  # today's data bar sits 2 sessions before the CPI session

px = close_panel(["SVXY", "SPY"])
all_dates = px.index
pos = pd.Series(np.arange(len(all_dates)), index=all_dates)
TDOM = tdom_of(all_dates)
ev = load_events(["cpi"])

svxy = px["SVXY"].dropna()
print(f"SVXY series: {svxy.index[0].date()} .. {svxy.index[-1].date()}  "
      f"({len(svxy)} bars)")
print(f"leverage break used: {SVXY_LEV_BREAK.date()}  "
      f"(-1x before, -0.5x on and after)")

svxy_h = rebase_half(svxy)  # constant -0.5x synthetic

anch = declusters(anchor_dates(ev, "cpi", OFFSET, all_dates), 5, all_dates)
anch = anch[anch.isin(svxy.index)]
pre = anch[anch < SVXY_LEV_BREAK]
post = anch[anch >= SVXY_LEV_BREAK]
print(f"\nCPI anchors usable on SVXY: N={len(anch)}  "
      f"pre-break {len(pre)} ({100*len(pre)/len(anch):.0f}%)  "
      f"post-break {len(post)}")
print(f"first CPI event in macro_events: "
      f"{load_events(['cpi'])['date'].min().date()}  -> "
      f"{(load_events(['cpi'])['date'] < svxy.index[0]).sum()} CPI prints "
      f"predate SVXY entirely and cannot be in the sample")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("A. reproduce the recon cell, then rerun on the constant -0.5x basis")
print("=" * 78)
rows = []
for label, s in (("RAW SVXY (as recon)", svxy), ("-0.5x CONSTANT", svxy_h)):
    for h in HORIZONS:
        f = fwd_lag(s, h, lag=1)
        v = f.reindex(anch).dropna()
        ctl = tdom_control(f, anch, TDOM, all_dates, pos)
        st = summarize(v.values, f"{label} h={h}")
        st["ctl_pct"] = 100 * ctl.mean()
        st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
        st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
        rows.append(st)
show(rows, "A1. recon reproduction vs leverage-neutral rerun")

print("\nA2. era split, RAW vs constant -0.5x (episodes, h=3)")
for label, s in (("RAW SVXY", svxy), ("-0.5x CONSTANT", svxy_h)):
    f = fwd_lag(s, 3, lag=1)
    out = []
    for era, m in (("-1x era (pre 2018-02)", anch < SVXY_LEV_BREAK),
                   ("-0.5x era (2018-02+)", anch >= SVXY_LEV_BREAK)):
        a = anch[m]
        v = f.reindex(a).dropna()
        ctl = tdom_control(f, a, TDOM, all_dates, pos)
        st = summarize(v.values, f"{label} | {era}")
        st["ctl_pct"] = 100 * ctl.mean()
        st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
        st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
        out.append(st)
    show(out, f"  {label}")

print("\nA3. the LIVE expectation: -0.5x era ONLY (post 2018-02-28), all horizons")
out = []
for h in HORIZONS:
    f = fwd_lag(svxy, h, lag=1)
    v = f.reindex(post).dropna()
    ctl = tdom_control(f, post, TDOM, all_dates, pos)
    st = summarize(v.values, f"2018-02+ h={h}")
    st["ctl_pct"] = 100 * ctl.mean()
    st["excess_pct"] = st["mean_pct"] - st["ctl_pct"]
    st["signp"] = sign_test(int((v.values > 0).sum()), len(v))
    out.append(st)
show(out, "A3. post-break only (this is what a 2026 order can expect)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("B. registry collision: decompose C1's window into the dead segment")
print("=" * 78)
ohlc = load_prices(["SVXY"])["SVXY"]
o, c = ohlc["Open"], ohlc["Close"]
cpi_sess = event_sessions(ev, "cpi", all_dates)
cpi_sess = cpi_sess[cpi_sess.isin(svxy.index)]

seg_rows = []
recs = []
for d in cpi_sess:
    p = pos.get(d)
    if p is None or p - 1 < 0 or p + 2 >= len(all_dates):
        continue
    prev, dd = all_dates[p - 1], d
    d2 = all_dates[p + 2]
    if not all(x in c.index for x in (prev, dd, d2)):
        continue
    if np.isnan(o.get(dd, np.nan)) or np.isnan(c.get(prev, np.nan)):
        continue
    on = o[dd] / c[prev] - 1.0                    # overnight INTO the print
    dead = c[d2] / o[dd] - 1.0                    # the registry's dead cell
    whole = c[d2] / c[prev] - 1.0                 # C1 h=3 (prev close -> +2)
    recs.append(dict(date=dd, overnight=on, dead_cell=dead, whole=whole))
seg = pd.DataFrame(recs).set_index("date")
# decluster to one per CPI (they are monthly already, but be strict)
seg = seg.loc[declusters(seg.index, 5, all_dates)]
for era, m in (("full 2011-10+", seg.index == seg.index),
               ("-1x era", seg.index < SVXY_LEV_BREAK),
               ("-0.5x era", seg.index >= SVXY_LEV_BREAK)):
    sub = seg[m]
    if not len(sub):
        continue
    seg_rows.append({
        "era": era, "n": len(sub),
        "overnight_bps": 1e4 * sub["overnight"].mean(),
        "dead_cell_bps": 1e4 * sub["dead_cell"].mean(),
        "whole_window_bps": 1e4 * sub["whole"].mean(),
        "dead_share_pct": 100 * sub["dead_cell"].mean() / sub["whole"].mean(),
        "overnight_t": sub["overnight"].mean() /
                       (sub["overnight"].std(ddof=1) / np.sqrt(len(sub))),
        "dead_t": sub["dead_cell"].mean() /
                  (sub["dead_cell"].std(ddof=1) / np.sqrt(len(sub))),
    })
show(seg_rows, "B1. C1 h=3 = [overnight into print] + [registry's dead cell]")

print("\nB2. the registry's exact cell, reproduced on real SVXY "
      "(CPI open -> +2 close)")
out = []
for era, m in (("full", seg.index == seg.index),
               ("2011-2017", seg.index < pd.Timestamp("2018-01-01")),
               ("2018+", seg.index >= pd.Timestamp("2018-01-01")),
               ("2021+", seg.index >= pd.Timestamp("2021-01-01"))):
    sub = seg[m]
    st = summarize(sub["dead_cell"].values, era)
    st["signp"] = sign_test(int((sub["dead_cell"].values > 0).sum()), len(sub))
    out.append(st)
show(out, "B2. sweep said: full +98 t3.3 / 2011-17 +177 t3.4 / 2018+ +39 t1.3")

print("\nB3. same era cut on the WHOLE C1 window (prev close -> +2 close)")
out = []
for era, m in (("full", seg.index == seg.index),
               ("2011-2017", seg.index < pd.Timestamp("2018-01-01")),
               ("2018+", seg.index >= pd.Timestamp("2018-01-01")),
               ("2021+", seg.index >= pd.Timestamp("2021-01-01"))):
    sub = seg[m]
    st = summarize(sub["whole"].values, era)
    st["signp"] = sign_test(int((sub["whole"].values > 0).sum()), len(sub))
    out.append(st)
show(out, "B3. if 2018+ and 2021+ collapse the same way, C1 IS the dead cell")

print("\nB4. and the overnight leg on its own, by era "
      "(the ONLY thing C1 adds to the dead cell)")
out = []
for era, m in (("full", seg.index == seg.index),
               ("2011-2017", seg.index < pd.Timestamp("2018-01-01")),
               ("2018+", seg.index >= pd.Timestamp("2018-01-01")),
               ("2021+", seg.index >= pd.Timestamp("2021-01-01"))):
    sub = seg[m]
    st = summarize(sub["overnight"].values, era)
    st["signp"] = sign_test(int((sub["overnight"].values > 0).sum()), len(sub))
    out.append(st)
show(out, "B4. overnight into the CPI print")
