"""C2 round 1 -- PPI on the curve: TLT / IEF with the long end at its 52w floor.

Distinctness owed vs the watchlist entry ("Long TLT from the NFP close to +3td
with the long end at its 52w floor", midterm-dead +0.071% N=12 t=0.17). Same
instrument, same price gate, different event. So the test that matters is GATE
ATTRIBUTION in both directions:
  (i) PPI anchor with NO price gate  -> does the event do anything?
  (ii) price gate with NO event      -> does the state do anything?
  (iii) both                          -> does the interaction add?
If (iii) ~= (ii), C2 IS the watchlist entry with a different anchor and
inherits its kill.

Anchor: 2 td before the PPI session (lag=1 entry = the close before the print).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "^TNX", "SPY"])
px = px.dropna(subset=["TLT", "IEF"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events(["ppi", "cpi", "nfp"])


def anchor_mask(kind: str, offset: int = -2) -> pd.Series:
    d = ev[ev.event == kind]["date"]
    m = pd.Series(False, index=idx)
    for x in d:
        p = int(idx.searchsorted(x, side="left"))
        if 0 <= p + offset < len(idx):
            m.iloc[p + offset] = True
    return m


m_ppi = anchor_mask("ppi")
m_cpi = anchor_mask("cpi")
m_nfp = anchor_mask("nfp")
print(f"PPI anchors in TLT era: {int(m_ppi.sum())}  CPI {int(m_cpi.sum())}  "
      f"NFP {int(m_nfp.sum())}   span {idx[0].date()}..{idx[-1].date()}")

# ---- price state: long end at its 52w floor -------------------------------
lo52 = px["TLT"].rolling(252).min()
dist_low = px["TLT"] / lo52 - 1.0          # today 0.0103
print(f"TODAY dist_52w_low = {100*dist_low.iloc[-1]:.2f}%")

gates = {
    "floor<=1.5%": dist_low <= 0.015,
    "floor<=3% (base)": dist_low <= 0.030,
    "floor<=5%": dist_low <= 0.050,
    "floor<=10%": dist_low <= 0.100,
}
for k, g in gates.items():
    print(f"  {k}: {int(g.sum())} days ({100*g.mean():.1f}% of history)")

base = gates["floor<=3% (base)"]

for H in (1, 3, 5):
    print("\n" + "#" * 78)
    print(f"HORIZON h={H}")
    print("#" * 78)

    print("\n--- (i) PPI anchor, NO price gate: does PPI move TLT at all? ---")
    battery(px, m_ppi, [("TLT", 1.0)], h=H, title=f"TLT long into PPI (ungated) h={H}",
            cost_bps=2.0, lag=1, min_gap=5, event_kinds=("cpi",))

    print("\n--- (ii) price gate ONLY, no event anchor ---")
    battery(px, base, [("TLT", 1.0)], h=H,
            title=f"TLT long at 52w floor, ANY day h={H}", cost_bps=2.0,
            lag=1, min_gap=10, event_kinds=("ppi",),
            variants=gates)

    print("\n--- (iii) BOTH: PPI anchor AND long end at the floor ---")
    both = m_ppi & base
    battery(px, both, [("TLT", 1.0)], h=H,
            title=f"TLT long into PPI at 52w floor h={H}", cost_bps=2.0,
            lag=1, min_gap=5, event_kinds=("cpi",),
            variants={k: (m_ppi & g) for k, g in gates.items()})

print("\n" + "=" * 78)
print("IEF version (belly instead of long end), h=3, gated")
print("=" * 78)
lo52i = px["IEF"].rolling(252).min()
gi = (px["IEF"] / lo52i - 1.0) <= 0.030
print(f"TODAY IEF dist_52w_low = {100*(px['IEF'].iloc[-1]/lo52i.iloc[-1]-1):.2f}%")
battery(px, m_ppi & gi, [("IEF", 1.0)], h=3,
        title="IEF long into PPI at 52w floor h=3", cost_bps=2.0, lag=1,
        min_gap=5, event_kinds=("cpi",))

print("\n" + "=" * 78)
print("MIDTERM SPLIT + head-to-head vs the watchlist's NFP form (h=3)")
print("=" * 78)
r3 = vehicle_ret(px, [("TLT", 1.0)], 3, 1)
allr = r3.dropna()
for lbl, m in [("PPI + floor", m_ppi & base), ("CPI + floor", m_cpi & base),
               ("NFP + floor (WATCHLIST)", m_nfp & base),
               ("floor only", base), ("PPI only", m_ppi)]:
    d = idx[m.values & r3.notna().values]
    d = declusters(d, 5, allr.index)
    if len(d) == 0:
        print(f"{lbl}: no episodes")
        continue
    v = r3.loc[d].values
    yrs = pd.DatetimeIndex(d).year
    mid = yrs % 4 == 2
    s = summarize(v, lbl)
    w = int((v > 0).sum())
    print(f"\n{lbl}: N={s['n']} mean {s['mean_pct']:+.3f}% hit {s['hit']:.0f}% "
          f"t {s['t']:+.2f} worst {s['worst_pct']:+.2f}% "
          f"| all-days ctrl {100*allr.mean():+.3f}% edge "
          f"{s['mean_pct']-100*allr.mean():+.3f}pp | sign p "
          f"{sign_test(w, len(v)):.4f} boot {bootstrap_p_le0(v):.3f}")
    show([summarize(v[mid], f"  midterm N={int(mid.sum())}"),
          summarize(v[~mid], f"  non-midterm N={int((~mid).sum())}")], "")
    print("  ", cluster_note(d, v))
