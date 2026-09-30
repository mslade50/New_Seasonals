"""sb1 GS round-2 follow-up: residual vs XLF/SPY against its UNCONDITIONAL residual drift, by era."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sb1_engine as E

import numpy as np
import pandas as pd

E.PX = load_prices(["GS", "XLF", "SPY", "MS", "JPM"])
px = pd.DataFrame({t: E.PX[t]["Close"] for t in E.PX}).dropna()
rets = px.pct_change()

# unconditional 21d residual (static full-sample beta from daily rets, same for window and control)
for bench in ("XLF", "SPY"):
    b = np.polyfit(rets[bench].dropna(), rets["GS"].dropna(), 1)[0]
    res_all = fwd_lag(px["GS"], 21, 2) - b * fwd_lag(px[bench], 21, 2)
    w = E.windows("GS", px, 2, 21)
    res_w = w.ret - b * w[bench.lower()]
    print(f"\n== GS resid vs {bench} (static beta {b:.2f}) ==")
    show([summarize(res_w.values, "Oct windows all"),
          summarize(res_w[w.index >= 2010].values, "Oct windows 2010+"),
          summarize(res_w[w.index < 2010].values, "Oct windows pre-2010"),
          summarize(res_w[w.index.isin(E.MID)].values, "Oct windows midterm"),
          summarize(res_all.dropna().values, "UNCOND all days"),
          summarize(res_all[res_all.index >= "2010-01-01"].dropna().values, "UNCOND 2010+")])
    k = int((res_w[w.index >= 2010] > 0).sum())
    print(f"  2010+ resid record {k}/{(w.index>=2010).sum()} sign p={sign_test(k, int((w.index>=2010).sum())):.3f}")

# raw 2010+ vs XLF 2010+ (is GS's clean-era edge just XLF's clean-era window?)
w = E.windows("GS", px, 2, 21)
m = w.index >= 2010
print(f"\n2010+ raw GS {100*w.ret[m].mean():+.2f}% ({int((w.ret[m]>0).sum())}/{m.sum()})  XLF {100*w.xlf[m].mean():+.2f}% "
      f"({int((w.xlf[m]>0).sum())}/{m.sum()})  SPY {100*w.spy[m].mean():+.2f}%  | uncond 21d 2010+: GS "
      f"{100*fwd_lag(px['GS'],21,2)['2010':].mean():+.2f}% XLF {100*fwd_lag(px['XLF'],21,2)['2010':].mean():+.2f}%")
for peer in ("MS", "JPM"):
    pw = E.windows(peer, px, 2, 21) if False else None
    pr = [px[peer].values[int(r.e1)] / px[peer].values[int(r.e0)] - 1 for _, r in w.iterrows()]
    pr = pd.Series(pr, index=w.index)
    print(f"  peer {peer}: all {100*pr.mean():+.2f}% ({int((pr>0).sum())}/26), 2010+ {100*pr[m].mean():+.2f}%, "
          f"midterm {100*pr[pr.index.isin(E.MID)].mean():+.2f}%  uncond {100*fwd_lag(px[peer],21,2).mean():+.2f}%")
