"""Friday printed higher US yields AND a sharply stronger yen on the same
session. Those normally move together the other way.

Friday 2026-09-04: ^TNX +0.46%, ^FVX +0.91%, USDJPY -1.70%, DXY +0.16%.
Cell: US 10-year yield UP while USDJPY falls 1% or more, same session.
Then: does it matter that it was a payrolls session?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, fwd_ret, summarize, sign_test, era_split,  # noqa
                       cluster_note, declusters, local_control, load_events)

px = close_panel(["^TNX", "JPY=X", "^GSPC", "DX-Y.NYB", "SPY"])
tnx, jpy, spx, dxy = (px[c].dropna() for c in ("^TNX", "JPY=X", "^GSPC", "DX-Y.NYB"))
idx = px.index

dt = tnx.pct_change()
dj = jpy.pct_change()
dd = dxy.pct_change()
print(f"Friday 2026-09-04: TNX {100*dt.iloc[-1]:+.2f}%  USDJPY {100*dj.iloc[-1]:+.2f}%  "
      f"DXY {100*dd.iloc[-1]:+.2f}%")
print(f"  correlation of daily TNX and USDJPY moves, full history: "
      f"{dt.corr(dj):+.2f}; last 252 sessions: {dt.tail(252).corr(dj.tail(252)):+.2f}")


def line(label, dates, s, h):
    d = pd.DatetimeIndex([x for x in dates if x in s.index])
    r = fwd_ret(s, h).reindex(d).dropna()
    if len(r) < 3:
        print(f"  {label:46} h{h:<3} n={len(r)} thin"); return None
    v = r.values; up = int((v > 0).sum()); st = summarize(v, label)
    print(f"  {label:46} h{h:<3} n={len(v):5d} mean={st['mean_pct']:+7.3f}% "
          f"med={st['median_pct']:+7.3f}% {up}-{len(v)-up} hit={st['hit']:5.1f}% "
          f"t={st['t']:+5.2f} signp={sign_test(up, len(v)):.4f}")
    return r


m = (dt > 0) & (dj <= -0.01)
trig = pd.DatetimeIndex([d for d in m.index[m.fillna(False)] if d < idx[-1]])
dec = declusters(trig, 5, idx)
print(f"\n10y yield up while USDJPY falls 1%+: {len(trig)} sessions, "
      f"{len(dec)} declustered at 5 td")
print(f"  by year: {pd.Series([d.year for d in dec]).value_counts().sort_index().to_dict()}")

print("\n=== forward ===")
for h in (1, 5, 10, 21):
    line("USDJPY", dec, jpy, h)
for h in (1, 5, 21):
    line("^GSPC", dec, spx, h)
for h in (1, 5, 21):
    line("^TNX", dec, tnx, h)

print("  controls")
ctrl = local_control(idx, dec, 126)
for h in (1, 5, 21):
    line("  ^GSPC local +/-126td", ctrl, spx, h)
    line("  USDJPY local +/-126td", ctrl, jpy, h)

r5 = fwd_ret(spx, 5).reindex(pd.DatetimeIndex([d for d in dec if d in spx.index])).dropna()
print("\n  ^GSPC h5 era:", [f"{e['label']} n={e['n']} mean={e['mean_pct']:+.2f}% hit={e['hit']:.1f}%"
                            for e in era_split(r5.index, r5.values)])
print("  ^GSPC h5 concentration:", cluster_note(r5.index, r5.values, 2))

print("\n=== was it a payrolls session? ===")
try:
    ev = load_events(["nfp"])
    nfp = pd.DatetimeIndex(pd.to_datetime(ev["date"]))
    on_nfp = pd.DatetimeIndex([d for d in dec if d in set(nfp)])
    off_nfp = pd.DatetimeIndex([d for d in dec if d not in set(nfp)])
    print(f"  of {len(dec)} episodes, {len(on_nfp)} landed on a payrolls session")
    print(f"  payrolls episodes: {[str(d.date()) for d in on_nfp]}")
    for h in (1, 5):
        line("USDJPY, payrolls sessions only", on_nfp, jpy, h)
        line("^GSPC, payrolls sessions only", on_nfp, spx, h)
    for h in (1, 5):
        line("USDJPY, non-payrolls", off_nfp, jpy, h)
except Exception as exc:
    print("  events unavailable:", exc)
