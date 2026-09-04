"""The publishable leg from 04 is h=2, the print day itself: ^VIX fell on
201 of 319 payroll sessions. Era-split it, control it properly, and check
whether the calm-going-in conditioning survives both halves.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^VIX", "SPY"])
vx = px["^VIX"]["Close"].dropna()

ev = load_events(["nfp"])["date"]
pos, kept = anchor_positions(vx.index, ev, offset=-2)
anch = vx.index[pos]
anch = anch[anch < vx.index[-1]]

r2 = fwd_ret(vx, 2)          # anchor -> print day close
own = vx.pct_change()        # every session's own move


def rec(v, lab):
    d = summarize(v.values, lab)
    dn = int((v < 0).sum())
    d["up"], d["down"] = len(v) - dn, dn
    d["pct_down"] = round(100 * dn / len(v), 1) if len(v) else None
    d["sign_p_down"] = round(sign_test(dn, len(v)), 4) if len(v) else None
    return d


v = r2.reindex(anch).dropna()
rows = [rec(v, "print day, all anchors")]
for lab, m in [("pre-2018", v.index < pd.Timestamp("2018-01-01")),
               ("2018+", v.index >= pd.Timestamp("2018-01-01")),
               ("midterm years", (v.index.year % 4) == 2),
               ("September anchors", v.index.month == 9)]:
    rows.append(rec(v[np.asarray(m)], lab))

# controls: what a random session does, and what a random 2-day span does
rows.append(rec(own.dropna(), "ctl: every VIX session"))
r2_all = r2.dropna()
rows.append(rec(r2_all, "ctl: every 2-session span"))
show(rows, "^VIX from the k=2 anchor close to the payrolls close")
print(cluster_note(v.index, v.values, k=2))

# --- conditioned on calm going in, both eras ---
rk = pct_rank(vx, 63)
lvl = rolling_on_valid(vx, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"\nlive: 63d return rank {rk.iloc[-1]:.1f}, level rank {lvl.iloc[-1]:.1f}")

for cname, cond in [("63d return rank <= 25", rk.reindex(anch) <= 25),
                    ("level rank <= 40", lvl.reindex(anch) <= 40),
                    ("both", (rk.reindex(anch) <= 25) & (lvl.reindex(anch) <= 40))]:
    a = anch[cond.fillna(False).values]
    vv = r2.reindex(a).dropna()
    rows = [rec(vv, f"{cname} (n={len(vv)})")]
    for lab, m in [("  pre-2018", vv.index < pd.Timestamp("2018-01-01")),
                   ("  2018+", vv.index >= pd.Timestamp("2018-01-01"))]:
        sub = vv[np.asarray(m)]
        if len(sub) >= 5:
            rows.append(rec(sub, lab))
    # matched control: same VIX state, no payroll in the next two sessions
    allc = vx.index[(rk <= 25).reindex(vx.index).fillna(False).values] if cname.startswith("63d") else None
    show(rows)
    print("   episodes:", [str(d.date()) for d in a][-8:])

# matched control done properly: same calm state, non-payroll anchors
mask = (rk <= 25) & (lvl <= 40)
cal = vx.index[mask.fillna(False).values]
cal = cal.difference(anch)
vv = r2.reindex(cal).dropna()
show([rec(vv, "ctl: same calm state, NOT a pre-payrolls anchor")],
     "matched control")
