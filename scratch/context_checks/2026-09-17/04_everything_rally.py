"""Today: SPY +1.13%, TLT +1.11%, VIX -12.82%. Everything rallied at once.

`P9:stocks_bonds_up` fires at a 50 bp threshold and reads SPY +0.034% next
session, edge -0.005%, i.e. nothing. That threshold is far too loose for what
printed. Tighten to 1% on BOTH legs, then add the vol collapse, and check
whether the rarer version says anything the loose one cannot.

There is no `VIX down 10%` trigger in PRICE_TRIGGERS at all (only the +10%
side), so the most characteristic feature of today's tape is computed here by
hand.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, cluster_note, declusters,
)

px = close_panel(["SPY", "TLT", "^VIX", "IWM", "QQQ"])
idx = px.index
spy, tlt, vix = px["SPY"].dropna(), px["TLT"].dropna(), px["^VIX"].dropna()
common = spy.index.intersection(tlt.index).intersection(vix.index)

rs = (spy / spy.shift(1) - 1.0).reindex(common)
rt = (tlt / tlt.shift(1) - 1.0).reindex(common)
rv = (vix / vix.shift(1) - 1.0).reindex(common)
f1 = (spy.shift(-1) / spy - 1.0).reindex(common)
f5 = (spy.shift(-5) / spy - 1.0).reindex(common)

print(f"TLT history starts {tlt.index[0].date()}; common sessions {len(common)}")

masks = [
    ("SPY +50bp & TLT +50bp (the engine cell)", (rs >= 0.005) & (rt >= 0.005)),
    ("SPY +1% & TLT +1%", (rs >= 0.01) & (rt >= 0.01)),
    ("SPY +1% & TLT +1% & VIX -8%", (rs >= 0.01) & (rt >= 0.01) & (rv <= -0.08)),
    ("SPY +1% & TLT +1% & VIX -10%", (rs >= 0.01) & (rt >= 0.01) & (rv <= -0.10)),
    ("SPY +1% & VIX -10% (no bond leg)", (rs >= 0.01) & (rv <= -0.10)),
    ("VIX -10% alone", rv <= -0.10),
]
for h, f in [(1, f1), (5, f5)]:
    rows = []
    for lab, m in masks:
        d = common[m.fillna(False).values]
        d = d.intersection(f.dropna().index)
        r = summarize(f.loc[d].values, lab)
        if r["n"]:
            k = int((f.loc[d] > 0).sum())
            r["record"] = f"{k}-{r['n'] - k} up"
            r["sign_p_up"] = round(sign_test(k, r["n"]), 4)
        rows.append(r)
    rows.append(summarize(f.dropna().values, "CTL all sessions"))
    show(rows, f"SPY h={h} after each condition")

print("\n=== detail on SPY +1% & TLT +1% & VIX -8% ===")
m = ((rs >= 0.01) & (rt >= 0.01) & (rv <= -0.08)).fillna(False)
d = common[m.values].intersection(f5.dropna().index)
tbl = pd.DataFrame({"date": [x.date() for x in d],
                    "spy_pct": (100 * rs.loc[d]).round(2).values,
                    "tlt_pct": (100 * rt.loc[d]).round(2).values,
                    "vix_pct": (100 * rv.loc[d]).round(2).values,
                    "spy_h1_pct": (100 * f1.loc[d]).round(2).values,
                    "spy_h5_pct": (100 * f5.loc[d]).round(2).values})
print(tbl.to_string(index=False))
v = f5.loc[d]
k = int((v > 0).sum())
print(f"  h5 record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
      f"median {100 * v.median():+.2f}%, sign p(up) = {sign_test(k, len(v)):.4f}")
print(f"  concentration: {cluster_note(d, v.values, k=2)}")

print("\n=== how many of these were the session after an FOMC decision? ===")
fomc = set(pd.DatetimeIndex(load_events(["fomc_decision"])["date"]))
pos = pd.Series(range(len(common)), index=common)
after = [x for x in d if pos[x] > 0 and common[pos[x] - 1] in fomc]
print(f"  {len(after)} of {len(d)}: {[str(x.date()) for x in after]}")

print("\n=== VIX itself after a 10%+ single-session collapse ===")
fv1 = (vix.shift(-1) / vix - 1.0).reindex(common)
fv5 = (vix.shift(-5) / vix - 1.0).reindex(common)
mv = (rv <= -0.10).fillna(False)
for h, f in [(1, fv1), (5, fv5)]:
    dd = common[mv.values].intersection(f.dropna().index)
    epi = declusters(dd, 5, common)
    rows = [summarize(f.loc[dd].values, f"VIX -10%+ day, h={h}"),
            summarize(f.loc[epi].values, f"declustered (5td), h={h}"),
            summarize(f.dropna().values, "CTL all sessions")]
    show(rows, f"VIX h={h}")
    k = int((f.loc[dd] > 0).sum())
    print(f"  record {k}-{len(dd) - k} up")
