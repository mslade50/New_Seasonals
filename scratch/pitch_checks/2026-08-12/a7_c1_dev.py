"""C1 round 3 (dev) -- the horizon has to come from this table, the entry form
has to be compared as WHOLE variants, and the losers have to name a number.

Cell: PPI print session with a CPI print on its eve. Anchor = 2 sessions before
the print; entry MOC on the eve close (lag=1); the print session is the hold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

raw = load_prices(["TLT"])["TLT"]
px = close_panel(["TLT"])
tl = px["TLT"].dropna()
idx = tl.index
N = len(idx)
c = tl.values
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
bh = float((d1[~np.isnan(d1)] > 0).mean())

ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 3 <= p < N - 12 and not np.isnan(d1[p])]
cpi_all = set(sp("cpi"))
live_p = [p for p in ppi_l if (p - 1) in cpi_all]
anchor = pd.DatetimeIndex([idx[p - 2] for p in live_p])   # lag=1 -> entry p-1
print(f"live-cell anchors (2 sessions before the print): N={len(anchor)}  "
      f"{anchor[0].date()}..{anchor[-1].date()}")

print("\n" + "=" * 100)
print("1. HORIZON SCAN from the anchor, lag=1 (entry MOC on the eve)")
print("   h=1 exits at the print close; h=2 holds one session past it, etc.")
print("=" * 100)
show(horizon_scan(px, anchor, [("TLT", 1.0)], hs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  lag=1, min_gap=10), "live cell, long TLT")
print("  the pitched horizon must be the argmax of edge_pct here, not an")
print("  assumption carried in from the watchlist entry.")

print("\n" + "=" * 100)
print("2. ENTRY FORM: MOC vs a close-anchored LIMIT, as WHOLE variants")
print("   (never a marginal-fill decomposition -- registry rule)")
print("=" * 100)
atr = pd.Series(wilder_atr(raw["High"], raw["Low"], raw["Close"], 14),
                index=raw.index)
rows = []
v_moc, hit_moc = [], []
for p in live_p:
    e = c[p - 1]
    x = c[p]
    v_moc.append(x / e - 1.0)
rr = np.array(v_moc)
w = int((rr > 0).sum())
rows.append({"variant": "MOC on the eve close", "fill_rate": 100.0,
             "N_signals": len(rr), "N_fills": len(rr),
             "cond_mean_pct": round(100 * rr.mean(), 4),
             "whole_mean_pct": round(100 * rr.mean(), 4),
             "hit": round(100 * w / len(rr), 1),
             "signp": round(sign_test(w, len(rr), bh), 4)})
for k in [0.10, 0.25, 0.50]:
    fills, whole = [], []
    for p in live_p:
        d_eve = idx[p - 1]
        d_prn = idx[p]
        if d_eve not in atr.index or d_prn not in raw.index:
            continue
        a = float(atr.loc[d_eve])
        if not np.isfinite(a):
            continue
        lim = c[p - 1] - k * a
        lo = float(raw.loc[d_prn, "Low"])
        if lo <= lim:
            r = c[p] / lim - 1.0
            fills.append(r)
            whole.append(r)
        else:
            whole.append(0.0)      # no fill = no trade = flat
    if not whole:
        continue
    fills = np.array(fills)
    whole = np.array(whole)
    wf = int((fills > 0).sum()) if len(fills) else 0
    rows.append({"variant": f"LIMIT close-{k:.2f}ATR, exit print MOC",
                 "fill_rate": round(100 * len(fills) / len(whole), 1),
                 "N_signals": len(whole), "N_fills": len(fills),
                 "cond_mean_pct": round(100 * fills.mean(), 4) if len(fills) else np.nan,
                 "whole_mean_pct": round(100 * whole.mean(), 4),
                 "hit": round(100 * wf / len(fills), 1) if len(fills) else np.nan,
                 "signp": round(sign_test(wf, len(fills), bh), 4) if len(fills) else np.nan})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  whole_mean_pct is the number that decides: it charges the unfilled")
print("  signals as the zero they are, so variants are comparable end to end.")

print("\n" + "=" * 100)
print("3. EXIT SENSITIVITY -- why time-only")
print("=" * 100)
print("  The hold is ONE session that contains a scheduled 08:30 ET release.")
print("  A stop inside it is a stop on the print reaction itself, which is the")
print("  event being bought; a target caps the only session with the edge.")
print("  Measured anyway, on the live cell:")
for tag, mult in [("stop 0.5 ATR", 0.5), ("stop 1.0 ATR", 1.0)]:
    out = []
    for p in live_p:
        d_eve, d_prn = idx[p - 1], idx[p]
        if d_eve not in atr.index or d_prn not in raw.index:
            continue
        a = float(atr.loc[d_eve])
        if not np.isfinite(a):
            continue
        e = c[p - 1]
        stop = e - mult * a
        lo = float(raw.loc[d_prn, "Low"])
        op = float(raw.loc[d_prn, "Open"])
        if lo <= stop:
            fill = min(stop, op)
            out.append(fill / e - 1.0)
        else:
            out.append(c[p] / e - 1.0)
    out = np.array(out)
    print(f"  {tag:14s} mean {100*out.mean():+.4f}%  hit "
          f"{100*(out>0).mean():.1f}%  vs time-only {100*rr.mean():+.4f}% / "
          f"{100*(rr>0).mean():.1f}%")

print("\n" + "=" * 100)
print("4. THE LOSING OBSERVATIONS -- what kills it, with a number")
print("=" * 100)
dts = pd.DatetimeIndex([idx[p] for p in live_p])
ordr = np.argsort(rr)
print("  worst 8 print sessions in the live cell:")
for i in ordr[:8]:
    p = live_p[i]
    d = idx[p]
    o = float(raw.loc[d, "Open"]) if d in raw.index else np.nan
    lo = float(raw.loc[d, "Low"]) if d in raw.index else np.nan
    e = c[p - 1]
    print(f"    {d.date()}  close-to-close {100*rr[i]:+.2f}%   "
          f"gap at the open {100*(o/e-1):+.2f}%   "
          f"worst intraday {100*(lo/e-1):+.2f}%   "
          f"CPI-day move {100*d1[p-1]:+.2f}%")
lose = rr[rr <= 0]
print(f"\n  losing observations: {len(lose)} of {len(rr)}  "
      f"mean {100*lose.mean():+.3f}%  worst {100*lose.min():+.2f}%")
print(f"  the whole cell's downside sd is {100*rr.std(ddof=1):.3f}% for one")
print(f"  session; a 1-sigma bad print is about "
      f"{100*(rr.mean()-rr.std(ddof=1)):+.2f}%")
gaps = []
for p in live_p:
    d = idx[p]
    if d in raw.index:
        gaps.append(float(raw.loc[d, "Open"]) / c[p - 1] - 1.0)
gaps = np.array(gaps)
print(f"  the OPEN carries {100*gaps.mean():+.4f}% of the "
      f"{100*rr.mean():+.4f}% (i.e. {100*gaps.mean()/rr.mean():.0f}% of the move")
print("  is the 08:30 release gapping in before an MOC exit can be placed).")
