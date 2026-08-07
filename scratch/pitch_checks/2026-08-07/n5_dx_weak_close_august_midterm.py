"""N5: long DX at an NFP close that closed WEAK, narrowed to August + midterm.

McKinley's cell, and it is genuinely new. N4 gated the dollar on TLT sitting at
its 52w floor and treated midterm as a SPLIT of that sample. This gates on DX's
own weak close and adds a month restriction. Different trigger, different
sample, and the direction is the inversion N4 hinted at (the SHORT was
wrong-signed in midterms, so the LONG is what is being asked about).

Built as a sample HIERARCHY on purpose. Today's registry lesson is to count
occurrences of the joint state before believing an edge, so each narrowing step
prints its N before its mean, and the broad cells are reported whether or not
the narrow one survives.

'Weak close' has no single definition, so four are tested and the spread
between them IS the result.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

RAW = load_prices(["DX-Y.NYB"])["DX-Y.NYB"]
DX = RAW.dropna(subset=["Close"]).copy()
CAL = DX.index
POS = pd.Series(range(len(CAL)), index=CAL)
EV = load_events()
NFP = [d for d in EV.loc[EV.event == "nfp", "date"] if d in POS.index]

O, H_, L, C = DX["Open"], DX["High"], DX["Low"], DX["Close"]
rng = (H_ - L).replace(0, np.nan)

DEFS = {
    "down on the day":        C < C.shift(1),
    "close < open":           C < O,
    "close in bottom 1/3":    (C - L) / rng < 0.3333,
    "down day AND bottom 1/3": (C < C.shift(1)) & ((C - L) / rng < 0.3333),
}

print(f"DX-Y.NYB bars: {len(DX)}  {CAL[0].date()} .. {CAL[-1].date()}")
print(f"NFP dates in the DX calendar: {len(NFP)} "
      f"({NFP[0].date()} .. {NFP[-1].date()})")
aug_mid_all = [d for d in NFP if d.month == 8 and d.year % 4 == 2]
print(f"\nAugust NFPs in midterm years, ALL of them: "
      f"{[str(d.date()) for d in aug_mid_all]}")
print("  ^ this is the ceiling on the sample before any weak-close gate")


def fwd(d, h):
    p = POS[d]
    return np.nan if p + h >= len(CAL) else C.iloc[p + h] / C.iloc[p] - 1.0


def cell(dates, h, label):
    v = np.array([fwd(d, h) for d in dates])
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"label": label, "n": 0}
    s = summarize(v, label)
    return s


for name, weak in DEFS.items():
    print("\n" + "=" * 100)
    print(f"WEAK-CLOSE DEFINITION: {name}")
    print(f"   fires on 2026-08-06 (most recent bar): {bool(weak.iloc[-1])}")
    print("=" * 100)

    tiers = {
        "1. all NFP": NFP,
        "2. all NFP + weak close": [d for d in NFP if bool(weak.get(d, False))],
        "3. August NFP (any year)": [d for d in NFP if d.month == 8],
        "4. midterm NFP (any month)": [d for d in NFP if d.year % 4 == 2],
        "5. August NFP + weak close": [d for d in NFP if d.month == 8
                                       and bool(weak.get(d, False))],
        "6. midterm NFP + weak close": [d for d in NFP if d.year % 4 == 2
                                        and bool(weak.get(d, False))],
        "7. August + midterm NFP": aug_mid_all,
        "8. THE CELL: Aug + midterm + weak": [d for d in aug_mid_all
                                              if bool(weak.get(d, False))],
    }
    print("\n--- sample hierarchy (N first, always) ---")
    for k, v in tiers.items():
        print(f"  {k:<38} N = {len(v)}")

    rows = []
    for k, dates in tiers.items():
        if not dates:
            rows.append({"label": k, "n": 0})
            continue
        for h in (3, 5):
            s = cell(dates, h, f"{k}  +{h}td")
            rows.append(s)
    show([r for r in rows if r.get("n", 0) > 0], "LONG DX forward returns")

    the_cell = tiers["8. THE CELL: Aug + midterm + weak"]
    if the_cell:
        print(f"\n  the exact cell's dates and +3td outcomes:")
        for d in the_cell:
            r = fwd(d, 3)
            print(f"    {d.date()}  {100*r:+.3f}%" if not np.isnan(r)
                  else f"    {d.date()}  n/a")

print("\n" + "=" * 100)
print("CONTROL: DX unconditional drift, so none of the above is read naked")
print("=" * 100)
for h in (3, 5):
    r = (C.shift(-h) / C - 1.0).dropna()
    print(f"  all days  +{h}td: mean {100*r.mean():+.4f}%  "
          f"median {100*r.median():+.4f}%  hit {100*(r>0).mean():.1f}%  N={len(r)}")
aug = [d for d in CAL if d.month == 8]
for h in (3, 5):
    v = np.array([fwd(d, h) for d in aug])
    v = v[~np.isnan(v)]
    print(f"  August days +{h}td: mean {100*v.mean():+.4f}%  "
          f"hit {100*(v>0).mean():.1f}%  N={len(v)}")
