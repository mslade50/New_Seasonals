"""Tomorrow-specific tail risk + cost sanity for C2 and C3.

C2 holds through an NVDA print at h>=2 and through Jackson Hole (Fri +3 td)
at h>=3. C3 holds through Jackson Hole at h>=3. Both need the tail quoted as
a number, not as prose.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

px = close_panel(["SMH", "SPY", "XLU", "TLT", "NVDA"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
EARN = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
EARN["date"] = pd.to_datetime(EARN["date"])
nv = EARN[EARN["ticker"] == "NVDA"]["date"].sort_values().unique()
P = pd.DatetimeIndex(sorted({idx[idx.searchsorted(pd.Timestamp(x))] for x in nv
                             if idx.searchsorted(pd.Timestamp(x)) < len(idx)}))
P = P[P <= idx[-1]]

print("=== C2 TAIL: SMH's REACTION-DAY move (close P -> close P+1) ===")
rc = px["SMH"].pct_change().shift(-1)          # return of the session after P
v = rc.reindex(P).dropna().values
nvv = px["NVDA"].pct_change().shift(-1).reindex(P).dropna().values
print(f"  SMH  N={len(v)}  mean {100*v.mean():+.3f}%  sd {100*v.std(ddof=1):.2f}%  "
      f"|move| mean {100*np.abs(v).mean():.2f}%  p05 {100*np.percentile(v,5):+.2f}%  "
      f"p01 {100*np.percentile(v,1):+.2f}%  worst {100*v.min():+.2f}%")
print(f"  NVDA N={len(nvv)} sd {100*nvv.std(ddof=1):.2f}%  worst {100*nvv.min():+.2f}%")
print(f"  2020+ SMH reaction-day sd "
      f"{100*rc.reindex(P[P>='2020-01-01']).dropna().std(ddof=1):.2f}%, "
      f"worst {100*rc.reindex(P[P>='2020-01-01']).dropna().min():+.2f}%")
print("  -> at h>=2 the pitched hold eats this full distribution with no stop; "
      "at h=1 it avoids it entirely, and h=1 is the cell that ranks 16 of 16 "
      "on the offset ladder.")

print("\n=== C2/C3 JACKSON HOLE inside the hold (Fri 2026-08-28 = +3 td) ===")
ev = load_events(["jackson_hole"])["date"]
jh = pd.DatetimeIndex([idx[idx.searchsorted(pd.Timestamp(d))] for d in ev
                       if idx.searchsorted(pd.Timestamp(d)) < len(idx)])
for t in ["SMH", "XLU", "SPY"]:
    r = px[t].pct_change().reindex(jh).dropna()
    print(f"  {t} on the speech session: N={len(r)} mean {100*r.mean():+.3f}% "
          f"sd {100*r.std(ddof=1):.2f}% worst {100*r.min():+.2f}%  "
          f"| midterm-year subset mean "
          f"{100*r[[d.year%4==2 for d in r.index]].mean():+.3f}% "
          f"(N={sum(d.year%4==2 for d in r.index)})")

print("\n=== COST SANITY (bar = 5x round trip) ===")
for lbl, edge_bps, legs, bps in [
        ("C2 h=1 pitched (PIT<=10)", -26.1, 1, 5.0),
        ("C2 h=3 pitched (PIT<=25)", 102.1, 1, 5.0),
        ("C2 h=3 pitched, drop-top2", 14.0, 1, 5.0),
        ("C2 h=3 Aug x 2020+ (live)", -104.1, 1, 5.0),
        ("C3 h=3 pitched", 9.0, 1, 5.0),
        ("C3 h=3 pitched vs all-days edge", -4.2, 1, 5.0),
        ("C3 h=3 LIVE (SPY<=2% off hi)", -106.0, 1, 5.0),
        ("C3 h=5 pitched", 34.0, 1, 5.0)]:
    rt = legs * bps
    print(f"  {lbl:36s} {edge_bps:+7.1f} bps / {rt:.0f} bps rt = {edge_bps/rt:+6.1f}x "
          f"{'PASS' if edge_bps/rt >= 5 else 'FAIL'}")
print("  (the only C2 line clearing 5x is the h=3 cell whose top-2 episodes are "
      "88% of its total and whose 2018+ half is -1.137%)")
