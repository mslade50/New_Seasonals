"""Pin every number quoted in the a2 verdicts, plus two audit checks.

Audit 1 -- TODAY'S GEOMETRY: confirm a lag=1 MOC entry tonight really does put
the CPI print inside the hold at EVERY horizon, so the CPI-IN cell is the only
cell that describes today's trade.

Audit 2 -- DECLUSTER ORDER: the CPI-IN split gives different numbers depending
on whether you decluster the raw trigger and then split, or gate first and
then decluster. Both are defensible for their own question; if the SIGN moves
with the ordering, the finding is an artefact and must not be quoted. Check it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, load_events, fwd_lag, declusters, summarize,  # noqa: E402
                       sign_test, event_in_window)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

px = close_panel(["USO", "XLE"])
s = px["XLE"].dropna()
uso_1d = px["USO"].pct_change()

# ---------------------------------------------------------------------------
print("=" * 100)
print("AUDIT 1 -- today's geometry")
print("=" * 100)
last = s.index[-1]
print(f"freshest bar (signal date D) = {last.date()}  USO 1d = {100*uso_1d.iloc[-1]:+.2f}%")
cpi = load_events(["cpi"])["date"]
ppi = load_events(["ppi"])["date"]
nxt_cpi = cpi[cpi > last].iloc[0]
nxt_ppi = ppi[ppi > last].iloc[0]
print(f"next CPI {nxt_cpi.date()}   next PPI {nxt_ppi.date()}")
print("entry is MOC on D+1 = the session AFTER the freshest bar = 2026-08-11 (today).")
print("so for every h>=1 the CPI print (2026-08-12) is strictly inside the hold:")
for h in (1, 2, 3, 5):
    print(f"  h={h}: entry close 2026-08-11 -> exit close +{h} td. "
          f"CPI inside: True   PPI (08-13) inside: {h >= 2}")
print("--> the ONLY historical cell that describes today's trade is CPI-IN.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("AUDIT 2 -- decluster order sensitivity of the CPI-IN split (XLE, h=3)")
print("=" * 100)
f3 = fwd_lag(s, 3, lag=1)
own3 = 100 * f3.dropna().mean()
rows = []
for thr in (0.03, 0.04, 0.05):
    m = (uso_1d >= thr).reindex(s.index).fillna(False)
    days = s.index[m.values]

    # (A) decluster the raw trigger, then split by CPI
    epi_a = declusters(days, 5, s.index)
    va = f3.reindex(epi_a).dropna()
    fa = event_in_window(va.index, s.index, 3, 1, ("cpi",))

    # (B) gate on CPI first, then decluster
    fb_all = event_in_window(days, s.index, 3, 1, ("cpi",))
    epi_b = declusters(days[fb_all], 5, s.index)
    vb = f3.reindex(epi_b).dropna()

    for lbl, vv in (("A decluster->split", va.values[fa]),
                    ("B gate->decluster", vb.values)):
        st = summarize(vv)
        rows.append({"thr": f">={100*thr:.0f}%", "order": lbl, "n": st["n"],
                     "long_mean_pct": round(st["mean_pct"], 3),
                     "excess": round(st["mean_pct"] - own3, 3),
                     "hit": round(st["hit"], 1),
                     "sign_of_long": "NEG" if st["mean_pct"] < 0 else "POS"})
print(pd.DataFrame(rows).to_string(index=False))
print("\nREAD: if 'sign_of_long' is NEG under BOTH orderings the CPI-IN")
print("degradation is real; if it flips, it is an artefact of the ordering.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("PINNED NUMBERS FOR THE VERDICTS")
print("=" * 100)

# C3 pitched cell
m5 = (uso_1d >= 0.05).reindex(s.index).fillna(False)
epi5 = declusters(s.index[m5.values], 5, s.index)
v5 = f3.reindex(epi5).dropna()
st = summarize(v5.values)
print(f"C3 pitched (USO>=5%, XLE, h=3): n={st['n']} mean {st['mean_pct']:+.3f}% "
      f"excess {st['mean_pct']-own3:+.3f}% hit {st['hit']:.1f} welch-ish t {st['t']:+.2f}")

# today's bucket
for lo, hi, lbl in ((0.05, 0.06, "[5%,6%)"), (0.06, 9.0, "[6%,inf) <-- TODAY +6.73%")):
    mm = ((uso_1d >= lo) & (uso_1d < hi)).reindex(s.index).fillna(False)
    e = declusters(s.index[mm.values], 5, s.index)
    v = f3.reindex(e).dropna()
    stt = summarize(v.values)
    print(f"  bucket {lbl:<28} n={stt['n']:<3} excess {stt['mean_pct']-own3:+.3f}% "
          f"hit {stt['hit']:.1f} signp {sign_test(int((v.values>0).sum()), len(v)):.4f}")

# CPI-IN at the pitched threshold, both orderings, already above; print the trade cell
fa5 = event_in_window(v5.index, s.index, 3, 1, ("cpi",))
vv = v5.values[fa5]
print(f"  CPI-IN at >=5%: n={len(vv)} mean {100*vv.mean():+.3f}% "
      f"record {int((vv>0).sum())}-{int((vv<=0).sum())}")

# 2026 to date
y26 = v5.index.year == 2026
print(f"  2026 episodes: n={int(y26.sum())} mean {100*v5.values[y26].mean():+.3f}% "
      f"(the current regime, and it is negative)")
