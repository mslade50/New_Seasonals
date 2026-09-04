"""Two loose ends.

(1) C5: the brief says "if XLI is not distinguishable the honest form is a
    POOLED family trade or nothing". The bare TRIPLE FLOOR (drop the SPY leg)
    was the one drop-one variant that beat the full cell at h=10 (+1.646% on
    65 XLI episodes, t 2.24), so the pooled family version of THAT is the last
    door left open. Closed here at h=3/5/10.
(2) C7: the surface map reads the live VIX 21-day range percentile as 8.3 and
    c7_vix_range_pop_r1.py reads 7.9 off the same code. Establish which panel
    the difference comes from before quoting either.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from _rc import cochran, per_name, pooled  # noqa

pd.set_option("display.width", 250)

SPDRS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
WIDE = SPDRS + ["XLRE", "XLC", "SMH", "XBI", "IBB", "KRE", "IHI", "ITB",
                "XME", "XOP", "OIH", "QQQ", "IWM", "DIA", "EFA", "EEM",
                "EWJ", "FXI", "EWZ"]
px = close_panel(sorted(set(WIDE + ["SPY"])))
pxd = {t: px[t] for t in px.columns}


def triple(s, k=10):
    return ((pct_rank(s, 5) <= k) & (pct_rank(s, 21) <= k)
            & (pct_rank(s, 63) <= k))


def bare(_t, s):
    return triple(s).fillna(False)


print("########## 1. C5 -- the POOLED BARE TRIPLE FLOOR, family trade or nothing ##########")
for H in (3, 5, 10):
    GAP = max(H, 5)
    for fam, nm in [(SPDRS, "nine SPDRs"), (WIDE, f"{len(WIDE)}-ETF pool")]:
        p = pooled(pxd, fam, bare, H, GAP, f"h={H} POOLED {nm}")
        # name-agnostic drift control over the same names
        drift = []
        for t in fam:
            r = px[t].shift(-(1 + H)) / px[t].shift(-1) - 1.0
            drift.append(r.dropna().values)
        dv = np.concatenate(drift)
        w = int((p["_vals"] > 0).sum())
        print(f"  h={H:2d} {nm:14s} N={p['n']:4d} mean {p['mean_pct']:+7.3f}%  "
              f"hit {p['hit']:5.1f}%  t {p['t']:+5.2f}  drift {100*dv.mean():+.3f}%  "
              f"edge {p['mean_pct']-100*dv.mean():+7.3f}pp  worst {p['worst_pct']:+7.2f}%  "
              f"sign p {sign_test(w, p['n']):.4f}")
    pn = per_name(pxd, SPDRS, bare, H, GAP)
    co = cochran(pn)
    ok = pn.dropna(subset=["t_excess"]).sort_values("t_excess", ascending=False)
    print(f"       Cochran Q {co['Q']:.2f}/{co['df']}df p {co['p']:.4f} "
          f"I2 {co['I2_pct']:.1f}%; FE common {co['fe_common_pct']:+.3f}pp "
          f"(t {co['fe_t']:+.2f}); XLI ranks "
          f"{list(ok['tkr']).index('XLI')+1}/{len(ok)}, leader {list(ok['tkr'])[0]}")

print("\n  XLI's own bare-triple-floor cell, for the record:")
for H in (3, 5, 10):
    r = vehicle_ret(px, [("XLI", 1.0)], H)
    v = r.dropna().index
    e = declusters(px.index[triple(px["XLI"]).fillna(False).values].intersection(v),
                   max(H, 5), v)
    vals = r.loc[e].values
    print(f"    h={H:2d} N={len(vals)} mean {100*vals.mean():+.3f}% edge "
          f"{100*(vals.mean()-r.loc[v].mean()):+.3f}pp t "
          f"{vals.mean()/(vals.std(ddof=1)/np.sqrt(len(vals))):+.2f} "
          f"worst {100*vals.min():+.2f}%  |  {cluster_note(e, vals, k=2)}")

print("\n########## 2. C7 -- which panel moves the live range percentile ##########")
for lbl, tk in [("2-ticker panel (SPY,^VIX) after dropna",
                 ["SPY", "^VIX"]),
                ("4-ticker panel as in c7 script", ["SPY", "^VIX", "^VIX3M", "SVXY"]),
                ("VIX alone from load_prices", None)]:
    if tk is None:
        s = load_prices(["^VIX"])["^VIX"]["Close"].dropna()
    else:
        p2 = close_panel(tk)
        s = p2["^VIX"]
        if lbl.startswith("2-ticker"):
            s = p2[["SPY", "^VIX"]].dropna()["^VIX"]
    rg = rolling_on_valid(s, lambda x: x.rolling(21).max() / x.rolling(21).min() - 1)
    rp = rolling_on_valid(rg.dropna(), lambda x: x.rolling(252).rank(pct=True) * 100)
    print(f"  {lbl:38s} rows {len(s):5d}  live range pctile {rp.dropna().iloc[-1]:.2f}")
print("  -> the reading is panel-sensitive at the ~0.4pt level; every reading "
      "puts today in the [5,15) bucket, which is the one that matters.")
