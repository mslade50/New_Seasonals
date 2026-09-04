"""BTC fired two independent BH-passing continuation cells, and the anchor
session contradicts both.

P4:z10_extreme|BTC-USD stretched up: n 298, h1 +0.772%, t 3.12, era stable.
P5b:rank21_extreme|BTC-USD top 5%: n 297, h1 +0.740%, t 2.85.
Both say a stretched bitcoin keeps going. But Friday itself was -3.63%, and
z10 is still +2.27 because the stretch is measured over ten sessions.

The base cell cannot answer the question that matters: when the stretch is
intact but the anchor session sold off hard, is the next session still a
continuation, or is the sell-off the turn? Split the cell on the anchor
session's own return.

Also worth knowing whether the cell is a bitcoin fact or a 2013-2017 fact,
since crypto history is short and front-loaded with parabolic regimes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
    declusters, local_control,
)

px = close_panel(["BTC-USD", "ETH-USD"])

for SUB in ["BTC-USD", "ETH-USD"]:
    s = px[SUB].dropna()
    r = s / s.shift(1) - 1.0
    # z10 exactly as build_pitch_state._metrics_for defines it: 10d return
    # over 21d vol scaled to 10d
    ret10 = s / s.shift(10) - 1.0
    vol21 = r.rolling(21).std()
    z10 = ret10 / (vol21 * np.sqrt(10))
    fwd1 = s.shift(-1) / s - 1.0
    fwd5 = s.shift(-5) / s - 1.0

    mask = z10 >= 2.0
    d = s.index[mask.fillna(False)]
    d = d[d <= s.index[-1]]

    print("=" * 78)
    print(f"{SUB}: z10 >= 2 (stretched up)")
    print("=" * 78)
    v1 = fwd1.reindex(d).values
    v5 = fwd5.reindex(d).values
    ok = np.isfinite(v1)
    d, v1, v5 = d[ok], v1[ok], v5[ok]
    up = int((v1 > 0).sum())
    base = summarize(fwd1.dropna().values, "all days")
    rows = [summarize(v1, "z10>=2, h1"), summarize(v5[np.isfinite(v5)], "z10>=2, h5"), base]
    rows[0]["record"] = f"{up}-{len(v1) - up}"
    show(rows, f"{SUB}: base cell")
    print("  era:", [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
                     for x in era_split(d, v1)])
    print("  conc:", cluster_note(d, v1))

    print()
    print("  --- the crossing: split on the ANCHOR session's own return ---")
    anchor_ret = r.reindex(d).values
    rows = []
    for label, m in [("anchor day <= -3%", anchor_ret <= -0.03),
                     ("anchor day -3% to 0", (anchor_ret > -0.03) & (anchor_ret <= 0)),
                     ("anchor day > 0", anchor_ret > 0)]:
        if m.sum() < 5:
            continue
        x = summarize(v1[m], label)
        u = int((v1[m] > 0).sum())
        x["record"] = f"{u}-{int(m.sum()) - u}"
        x["sign_p"] = round(sign_test(u, int(m.sum())), 4)
        rows.append(x)
    show(rows, f"{SUB}: continuation by how the anchor session closed")

    m = anchor_ret <= -0.03
    if m.sum() >= 5:
        print("  down-day arm era:",
              [(x["label"], x["n"], round(x["mean_pct"], 3), round(x["hit"], 1))
               for x in era_split(d[m], v1[m]) if x.get("n")])
        print("  down-day arm conc:", cluster_note(d[m], v1[m]))
        dd = declusters(pd.DatetimeIndex(d[m]), 5, s.index)
        keep = np.array([x in set(dd) for x in d[m]])
        if keep.sum() >= 5:
            x = summarize(v1[m][keep], "declustered 5td")
            u = int((v1[m][keep] > 0).sum())
            print(f"  declustered: n={x['n']} {u}-{x['n']-u} mean {x['mean_pct']:+.3f}% "
                  f"t={x['t']:+.2f}")
        print("  recent episodes:",
              [(str(pd.Timestamp(x).date()), round(100 * y, 2))
               for x, y in list(zip(d[m], v1[m]))[-8:]])
    print()

print("=" * 78)
print("today's readings")
print("=" * 78)
for SUB in ["BTC-USD", "ETH-USD"]:
    s = px[SUB].dropna()
    r = s / s.shift(1) - 1.0
    ret10 = s / s.shift(10) - 1.0
    z10 = ret10 / (r.rolling(21).std() * np.sqrt(10))
    print(f"  {SUB}: last {s.iloc[-1]:,.0f} on {s.index[-1].date()}  "
          f"session {100*r.iloc[-1]:+.2f}%  z10 {z10.iloc[-1]:+.2f}  "
          f"10d {100*ret10.iloc[-1]:+.2f}%")
