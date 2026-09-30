"""X1 round 1b: reference class + the neighbour that looked alive.

Reference class: the same one-day unwind rule on BRL, ZAR, TRY, AUDJPY, NZDJPY,
CADJPY (and MXN for the row). Threshold = each cross's own 95.3th pctile of the
adverse (carry-currency-down) daily move, i.e. the percentile USDMXN +1.25% sits
at, AND (^MOVE or ^VIX up). Also the fixed +1.25% rule. Long the carry currency
(USD-quoted crosses inverted). Spot only: carry is ~additive and does not move
the edge over own drift. Episodes declustered at 10 td.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

CR = {"MXN": ("USDMXN=X", True), "BRL": ("USDBRL=X", True), "ZAR": ("USDZAR=X", True),
      "TRY": ("USDTRY=X", True), "AUDJPY": ("AUDJPY=X", False), "NZDJPY": ("NZDJPY=X", False),
      "CADJPY": ("CADJPY=X", False)}
P = load_prices([v[0] for v in CR.values()] + ["^MOVE", "^VIX"])


def vchg(t, cal):
    s = P[t]["Close"].dropna()
    return (s / s.shift(1) - 1).reindex(cal)


rows = []
live = {}
for nm, (tk, inv) in CR.items():
    s = P[tk]["Close"].dropna()
    s = s[s > 0]
    # drop obviously broken prints (Yahoo FX glitches): |1d| > 15%
    lc = (1.0 / s) if inv else s
    d1 = lc.pct_change()
    bad = d1.abs() > 0.15
    lc = lc[~bad]
    d1 = lc.pct_change()
    cal = lc.index
    px = pd.DataFrame({nm: lc})
    volup = (vchg("^MOVE", cal) > 0) | (vchg("^VIX", cal) > 0)
    thr = d1.dropna().quantile(1 - 0.953)
    for rule, m in (("pctile", (d1 <= thr) & volup), ("fixed-1.25%", (d1 <= -0.0125) & volup),
                    ("pctile NO vol gate", d1 <= thr)):
        for h in (2, 5, 10):
            r = fwd_lag(px[nm], h)
            d = cal[m.fillna(False).values & r.notna().values]
            e = declusters(d, 10, cal)
            v = r.loc[e].values
            base = r.dropna()
            w = int((v > 0).sum())
            ph = float((base > 0).mean())
            row = summarize(v, f"{nm} {rule} h={h}")
            row.update({"thr_pct": round(100 * thr, 2), "own_pct": round(100 * base.mean(), 3),
                        "edge_pct": round(100 * (v.mean() - base.mean()), 3),
                        "rec": f"{w}-{len(v)-w}", "p_vs_base": round(sign_test(w, len(v), ph), 3)})
            e18 = e[e >= "2018-01-01"]
            row["edge18_pct"] = round(100 * (r.loc[e18].mean() - base.mean()), 3) if len(e18) else np.nan
            row["n18"] = len(e18)
            rows.append(row)
    live[nm] = (round(100 * d1.iloc[-1], 2), round(100 * thr, 2), str(cal[-1].date()))

df = pd.DataFrame(rows)
cols = ["label", "n", "mean_pct", "own_pct", "edge_pct", "t", "hit", "rec", "p_vs_base", "n18",
        "edge18_pct", "worst_pct", "thr_pct"]
pd.set_option("display.width", 250)
for c in ("mean_pct", "t", "hit", "worst_pct"):
    df[c] = df[c].round(3)
print(df[cols].to_string(index=False))
print("\nlive last-bar carry-ccy move vs its pctile threshold:", live)

# pooled reference class (ex MXN), pctile rule with vol gate, per horizon: mean edge per cross
print("\npooled ex-MXN edge (mean of per-cross edges), pctile rule + vol gate:")
for h in (2, 5, 10):
    sub = df[df.label.str.contains(f"pctile h={h}$") & ~df.label.str.startswith("MXN")]
    print(f"  h={h}: crosses {len(sub)}, positive edge {int((sub.edge_pct > 0).sum())}, "
          f"mean edge {sub.edge_pct.mean():+.3f}pp, 2018+ mean edge {sub.edge18_pct.mean():+.3f}pp")
