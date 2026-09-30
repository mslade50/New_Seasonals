import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
from sb5_common import PX, IDX, H, round1_2, anchors, yearly, MIDTERMS

import numpy as np
import pandas as pd

res = round1_2("XLF")

print("\n### (e) split at the October bank kickoff (earliest JPM/C/WFC Oct print)")
ec = pd.read_parquet(Path(__file__).resolve().parents[3] / "data" / "earnings_calendar.parquet",
                     columns=["ticker", "date"])
ec["date"] = pd.to_datetime(ec["date"])
ec = ec[ec.ticker.isin(["JPM", "C", "WFC"]) & (ec.date.dt.month == 10) & ec.date.dt.day.between(8, 25)]
kick = ec.groupby(ec.date.dt.year).date.min()
print(" kickoffs:", {int(y): str(d.date()) for y, d in kick.items()})
c = PX["XLF"].values
cs = PX["SPY"].values
for lag in (1, 2):
    anc = anchors(IDX, 9, 30)
    rows = []
    for y, p in anc.items():
        if y not in kick.index:
            continue
        k = int(IDX.searchsorted(kick[y]))  # kickoff session (pre-mkt print)
        e, x = p + lag, p + lag + H
        if x >= len(c) or not (e < k - 1 < x):
            continue
        pre = c[k - 1] / c[e] - 1
        post = c[x] / c[k - 1] - 1
        pre_s = cs[k - 1] / cs[e] - 1
        post_s = cs[x] / cs[k - 1] - 1
        rows.append((y, pre, post, pre - pre_s, post - post_s, k - 1 - e, x - k + 1))
    d = pd.DataFrame(rows, columns=["y", "pre", "post", "pre_ex", "post_ex", "npre", "npost"]).set_index("y")
    print(f" lag{lag}: N={len(d)} pre-print mean {100*d.pre.mean():+.2f}% hit {int((d.pre>0).sum())}/{len(d)} "
          f"(avg {d.npre.mean():.1f}td) | post-print {100*d.post.mean():+.2f}% hit {int((d.post>0).sum())}/{len(d)} "
          f"(avg {d.npost.mean():.1f}td)")
    print(f"        excess pre {100*d.pre_ex.mean():+.2f}% hit {int((d.pre_ex>0).sum())}/{len(d)} | "
          f"excess post {100*d.post_ex.mean():+.2f}% hit {int((d.post_ex>0).sum())}/{len(d)}")
    for cut in (2010, 2018):
        a = d[d.index >= cut]
        print(f"        {cut}+: pre {100*a.pre.mean():+.2f}% post {100*a.post.mean():+.2f}% | "
              f"ex pre {100*a.pre_ex.mean():+.2f}% ex post {100*a.post_ex.mean():+.2f}% (n={len(a)})")
    # control: XLF per-day drift x segment length
    dd = PX["XLF"].pct_change().mean()
    print(f"        per-td XLF drift {100*dd:.3f}% -> pre ctrl {100*dd*d.npre.mean():+.2f}% post ctrl {100*dd*d.npost.mean():+.2f}%")
