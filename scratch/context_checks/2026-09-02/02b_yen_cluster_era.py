"""Era stability and concentration for the 4+ cross 2-ATR-down cluster.

Half the 16 episodes are 2007-2011. If the cell is a crisis-era artifact it
does not publish.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

CROSSES = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "NZDJPY=X", "AUDJPY=X", "JPY=X"]
px = load_prices(CROSSES)


def atr_wilder(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


F = pd.DataFrame({t: (px[t]["Close"] - px[t]["Close"].shift(1))
                  <= -2.0 * atr_wilder(px[t]).shift(1) for t in CROSSES})
F = F.astype(float).fillna(0.0)
panel = close_panel(CROSSES + ["SPY"])
cnt = F.sum(axis=1).reindex(panel.index).fillna(0)
trig = panel.index[cnt >= 4]
trig = trig[trig < panel.index[-1]]
epi = declusters(trig, 5, panel.index)

for sub in ["JPY=X", "EURJPY=X"]:
    r1 = fwd_ret(panel[sub], 1)
    v = r1.reindex(epi).dropna()
    rows = []
    for lab, m in [("all", v.index == v.index),
                   ("pre-2018", v.index < pd.Timestamp("2018-01-01")),
                   ("2018+", v.index >= pd.Timestamp("2018-01-01")),
                   ("ex 2007-2011", ~v.index.year.isin(range(2007, 2012)))]:
        vv = v[m]
        s = summarize(vv.values, lab)
        u = int((vv > 0).sum())
        s["up"], s["down"] = u, len(vv) - u
        s["sign_p"] = round(sign_test(u, len(vv)), 4) if len(vv) else None
        rows.append(s)
    show(rows, f"{sub} h=1 after a 4+ cross cluster")
    print(cluster_note(v.index, v.values, k=2))
    print("  every episode:")
    for d, x in v.items():
        print(f"    {d.date()}  {100*x:+.2f}%")
    print()

# where does it sit across horizons
print(horizon_scan.__doc__.splitlines()[0])
rows = []
for h in (1, 2, 3, 5, 10):
    r = fwd_ret(panel["JPY=X"], h)
    v = r.reindex(epi).dropna()
    s = summarize(v.values, f"h={h}")
    base = r.dropna()
    s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
    u = int((v > 0).sum())
    s["up"], s["down"], s["sign_p"] = u, len(v) - u, round(sign_test(u, len(v)), 4)
    rows.append(s)
show(rows, "USDJPY horizon scan from the cluster session (lag 0)")
