import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
px = close_panel(["GLD", "GC=F", "^TNX"])
tnx = px["^TNX"].dropna()
r5 = pct_rank(tnx, 5)
hi252 = tnx.rolling(252).max()
hi63 = tnx.rolling(63).max()
# pooled "thrust week" union: any of the neighbour forms (one honest broad parent)
U = ((r5 >= 90) & (tnx >= hi63 - 1e-9)) | (r5 >= 95) | ((r5 >= 90) & (tnx >= 0.98 * hi252))

fomc = load_events(["fomc_decision"])["date"]

for V in ["GLD", "GC=F"]:
    pv = px[[V]].dropna()
    idx = pv.index
    vr5 = pct_rank(pv[V], 5)
    # exact live offset: FOMC decision = signal + 3 td (entry + 2)
    pos, kept = anchor_positions(idx, fomc, offset=-3)
    exact = pd.Series(False, index=idx)
    exact.iloc[pos] = True
    u = U.reindex(idx).fillna(False).astype(bool)
    lo = (vr5 <= 30).fillna(False)
    for h in (3, 4, 5):
        ret = vehicle_ret(pv, [(V, 1.0)], h)
        valid = ret.dropna().index
        ex = exact.reindex(valid).values
        uu = u.reindex(valid).values
        ll = lo.reindex(valid).values
        rows = []
        groups = {
            "exact FOMC-3 anchor, ALL (null parent)": ex,
            "exact & thrust-union": ex & uu,
            "exact & NO thrust": ex & ~uu,
            "exact & thrust & GLD r5<=30 (LIVE state)": ex & uu & ll,
            "exact & thrust & GLD r5>30": ex & uu & ~ll,
            "exact & NO thrust & GLD r5<=30": ex & ~uu & ll,
        }
        vals = {}
        for k, m in groups.items():
            d = valid[m]
            v = ret.loc[d].values
            vals[k] = v
            r = summarize(v, k)
            if r["n"]:
                w = int((v > 0).sum())
                r["rec"] = f"{w}-{len(v)-w}"
                r["sign_p"] = round(sign_test(w, len(v)), 4)
            rows.append(r)
        rows.append(summarize(ret.loc[valid].values, "all days"))
        show(rows, f"{V} h={h} exact live FOMC offset (one anchor per meeting)")
        a, b = vals["exact & thrust-union"], vals["exact & NO thrust"]
        if len(a) > 1:
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            print(f"  thrust vs no-thrust inside the FOMC-3 anchor: {100*(a.mean()-b.mean()):+.3f}pp welch t {(a.mean()-b.mean())/se:+.2f}")
        a2 = vals["exact & thrust & GLD r5<=30 (LIVE state)"]
        if len(a2) > 1:
            se = np.sqrt(a2.var(ddof=1) / len(a2) + b.var(ddof=1) / len(b))
            print(f"  LIVE state vs no-thrust: {100*(a2.mean()-b.mean()):+.3f}pp welch t {(a2.mean()-b.mean())/se:+.2f}")
            d = valid[groups["exact & thrust & GLD r5<=30 (LIVE state)"]]
            print("  LIVE-state episodes:", [(str(x.date()), round(100*ret.loc[x], 2)) for x in d])
            mt = np.array([x.year % 4 == 2 for x in d])
            print(f"  midterm {100*a2[mt].mean() if mt.any() else np.nan:+.3f}% n={int(mt.sum())} | non {100*a2[~mt].mean() if (~mt).any() else np.nan:+.3f}% n={int((~mt).sum())}")
            show(era_split(d, a2), "  era split LIVE state")

# outlier check for the one positive neighbour (L5 GLD h=5 max +11.49%)
pv = px[["GLD"]].dropna()
ret5 = vehicle_ret(pv, [("GLD", 1.0)], 5)
print("\nGLD h=5 returns >= +8%:", [(str(d.date()), round(100*x, 2)) for d, x in ret5[ret5 >= 0.08].items()][:12])
