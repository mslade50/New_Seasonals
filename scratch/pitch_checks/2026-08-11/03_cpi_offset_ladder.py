"""Is the CPI EVE a real anchor, or is it month position?

Today's whole board leans on one anchor: enter MOC the session before a CPI
print. The definition-neighbour probe for an anchor is the OFFSET LADDER --
slide the entry session from 5 before the print to 3 after it and see whether
the number is a spike at the eve or a plateau across the neighbourhood.

A plateau means the "CPI eve" label is decoration on mid-month drift, which is
exactly the trap `d4b_vix_week_vs_monthpos.py` caught for VIX-expiry week. A
spike at offset -1 that decays either side is an event anchor.

Also splits the eve cell by whether the hold is entered BEFORE the print
(holds the event) or AFTER it (the registry's already-dead post-CPI cell), so
the distinctness claim is measured rather than asserted.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, load_events, fwd_lag, declusters, summarize, sign_test  # noqa: E402

warnings.filterwarnings("ignore")

TK = ["SVXY", "QQQ", "SPY", "^VIX", "GLD", "USO", "XLE"]
px = close_panel(TK)
idx = px.index
ev = load_events(["cpi"])
cpi = pd.DatetimeIndex(sorted(ev.loc[ev["event"] == "cpi", "date"].unique()))

# entry session at offset k relative to the print session:
#   k = -1  -> the eve (TODAY'S TRADE): holds the print
#   k =  0  -> the print session close: the registry's dead post-CPI cell
OFFSETS = list(range(-5, 4))


def entry_dates(k: int) -> pd.DatetimeIndex:
    out = []
    for d in cpi:
        loc = idx.searchsorted(d)
        if loc >= len(idx):
            continue
        j = loc + k
        if 0 <= j < len(idx):
            out.append(idx[j])
    return pd.DatetimeIndex(sorted(set(out)))


print("Entry at offset k from the CPI print session; MOC entry ON that session,")
print("so the mask date is k-1 and lag=1 lands the entry on k. h counted from entry.")
print("k=-1 is TODAY (holds the print). k=0 is the dead post-CPI cell.\n")

for tkr in TK:
    s = px[tkr].dropna()
    if len(s) < 500:
        continue
    print(f"===== {tkr}  ({s.index[0].date()} -> {s.index[-1].date()}) =====")
    print("   k   N     h1 mean   hit(  p  )   h3 mean   hit(  p  )")
    for k in OFFSETS:
        ent = entry_dates(k)
        # mask sits one session before the entry, lag=1 puts entry on `ent`
        mpos = idx.searchsorted(ent) - 1
        mpos = mpos[mpos >= 0]
        mask_dates = idx[mpos]
        mask_dates = declusters(mask_dates[mask_dates.isin(s.index)], 5, s.index)
        cells, n_shown = [], 0
        for h in (1, 3):
            v = fwd_lag(s, h, lag=1).reindex(mask_dates).dropna()
            if len(v) < 8:
                cells.append("     --       --        ")
                continue
            st = summarize(v.values)
            p = sign_test(int((v.values > 0).sum()), len(v))
            n_shown = st["n"]
            cells.append(f"{st['mean_pct']:+8.3f}  {st['hit']:5.1f}({p:.3f})  ")
        mark = "  <-- TODAY" if k == -1 else ("  <-- registry's dead post-CPI cell" if k == 0 else "")
        print(f"  {k:+d}  {n_shown:<4}" + "".join(cells) + mark)
    print()

# --- the distinctness question, stated as one number -----------------------
print("=" * 78)
print("DISTINCTNESS: eve entry (holds the print) vs print-close entry (post-CPI)")
print("=" * 78)
for tkr in ["SVXY", "^VIX"]:
    s = px[tkr].dropna()
    for k, label in ((-1, "eve  (holds print)"), (0, "print close (post)")):
        ent = entry_dates(k)
        mpos = idx.searchsorted(ent) - 1
        mask_dates = idx[mpos[mpos >= 0]]
        mask_dates = declusters(mask_dates[mask_dates.isin(s.index)], 5, s.index)
        out = [f"{tkr:>6} {label:<20}"]
        for h in (1, 2, 3, 5):
            v = fwd_lag(s, h, lag=1).reindex(mask_dates).dropna()
            if len(v) < 8:
                continue
            st = summarize(v.values)
            out.append(f"h{h} {st['mean_pct']:+6.2f}({st['hit']:4.1f}%)")
        print("  ".join(out))
    print()
