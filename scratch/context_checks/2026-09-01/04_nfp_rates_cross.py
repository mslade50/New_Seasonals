"""Pre-payrolls drift, crossed with the state that is actually live tonight.

The engine's E:nfp k=3 SPY cell: n=316, h1 +0.175%, 188-127, t 2.87, sign p
0.0004, BH pass, era-stable. This is a PRE-SPECIFIED famous cell (drift into
payrolls) so its BH pass is not what earns it a place. The question worth
asking is whether it survives the conditioning:

  A. ^TNX within 1% of a 252d high at the anchor (tonight: at it, 0.00%)
  B. midterm years
  C. September payrolls
  D. the ^VIX mirror

Anchor = the session 3 td before the print, so h1 = Wednesday, h2 = Thursday,
h3 = the payrolls session itself. lag=0 close-to-close, brief convention.

NOTE: pitch_lab.anchor_positions returns (positions, kept EVENT dates). The
anchor dates are idx[positions]. Getting that backwards measures forward
returns from the print itself and silently inverts the whole cell; it did
here on the first pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["SPY", "^GSPC", "IWM", "^VIX", "^TNX", "TLT"]
px = close_panel(TKRS).dropna(subset=["^GSPC"])
idx = px.index
ev = load_events(["nfp"])
ev = ev[(ev["date"] >= idx[0]) & (ev["date"] <= idx[-1])]
print(f"panel {idx[0].date()} .. {idx[-1].date()};  {len(ev)} payrolls prints")

pos3, kept3 = anchor_positions(idx, ev["date"], offset=-3)
anchors = idx[pos3]
ev_of = pd.Series(pd.DatetimeIndex(kept3).values, index=anchors)
print(f"anchors 3 td before a print: {len(anchors)}  "
      f"(first {anchors[0].date()}, last {anchors[-1].date()})")
print(f"  sanity: today is an anchor? {idx[-1] in set(anchors)} "
      f"-- next print {ev['date'].iloc[-1].date() if len(ev) else 'n/a'}")

tnx_hi = rolling_on_valid(px["^TNX"], lambda x: x.rolling(252, min_periods=200).max())
tnx_near = (px["^TNX"] >= tnx_hi * 0.99)
print(f"^TNX within 1% of a 252d high tonight: {bool(tnx_near.iloc[-1])} "
      f"({px['^TNX'].iloc[-1]:.3f} vs 252d high {tnx_hi.iloc[-1]:.3f})")


def cell(subj, dts, label, hs=(1, 2, 3, 5)):
    dts = pd.DatetimeIndex(dts)
    rows = []
    for h in hs:
        r = fwd_ret(px[subj], h)
        v = r.loc[r.index.intersection(dts)].dropna()
        s = summarize(v.values, f"h={h}")
        if s["n"]:
            w = int((v.values > 0).sum())
            s["record"] = f"{w}-{s['n']-w}"
            s["sign_p"] = round(sign_test(w, s["n"]), 4)
            s["edge_pct"] = round(s["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(s)
    show(rows, f"{subj}: {label}")


print("\n### A. the base cell, reproduced")
cell("SPY", anchors, f"all {len(anchors)} pre-payrolls anchors")

near_mask = tnx_near.reindex(anchors).fillna(False).astype(bool).values
sel = anchors[near_mask]
print(f"\n### B. anchors with ^TNX within 1% of a 252d high: {len(sel)}")
print("  " + ", ".join(str(d.date()) for d in sel))
cell("SPY", sel, "^TNX within 1% of a 252d high at the anchor")
cell("SPY", anchors[~near_mask], "^TNX NOT near a 252d high", hs=(1, 3))

print("\n### C. midterm years")
cell("SPY", anchors[anchors.year % 4 == 2],
     f"midterm-year payrolls ({int((anchors.year % 4 == 2).sum())} anchors)",
     hs=(1, 3, 5))

print("\n### D. September payrolls (the print lands in September)")
sep = anchors[pd.DatetimeIndex(ev_of.values).month == 9]
cell("SPY", sep, f"September payrolls ({len(sep)} anchors)", hs=(1, 3, 5))

print("\n### E. the ^VIX mirror")
cell("^VIX", anchors, "all anchors", hs=(1, 3))
cell("^VIX", sel, f"^TNX near a 252d high ({len(sel)})", hs=(1, 3))

print("\n### F. era split of the base cell")
for h in (1, 3):
    r = fwd_ret(px["SPY"], h)
    v = r.loc[r.index.intersection(anchors)].dropna()
    show(era_split(v.index, v.values), f"SPY h={h}, all anchors")

print("\n### G. concentration, h=1 and h=3")
for h in (1, 3):
    r = fwd_ret(px["SPY"], h)
    v = r.loc[r.index.intersection(anchors)].dropna()
    print(f"  h={h}: " + cluster_note(v.index, v.values, k=2))

print("\n### H. the shape of the run-in: which of the 3 sessions carries it")
r1 = fwd_ret(px["SPY"], 1)
for off, name in ((-3, "anchor -> +1 (2 td before the print)"),
                  (-2, "anchor -> +1 (1 td before the print)"),
                  (-1, "anchor -> +1 (the print session)")):
    p, _ = anchor_positions(idx, ev["date"], offset=off)
    a = idx[p]
    v = r1.loc[r1.index.intersection(a)].dropna()
    w = int((v.values > 0).sum())
    s = summarize(v.values, name)
    print(f"  {name:<42} n {s['n']:>3}  {w}-{s['n']-w}  "
          f"mean {s['mean_pct']:+.3f}%  t {s['t']:+.2f}  "
          f"sign p {sign_test(w, s['n']):.4f}")
