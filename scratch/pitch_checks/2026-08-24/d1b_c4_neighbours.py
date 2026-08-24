"""C4 round 2 -- definition neighbours and historical support.

Round 1 killed the pair on leg attribution (naked long +2.284% at h=5 against
pair +0.119% and a 1.5x cost multiple) and killed the outright on registry
collision (XLI's plain washout is a BELOW-median member of the book's own
dip-buy family, and 2018+ it is +0.074% at a 52.2% hit).

This round finishes it with the two things a kill report owes:
  (1) the neighbour ladder, NEAR neighbours included -- rank rung, peer
      proximity rung, and the r5 lookback -- because the gated cell is N=9
      days and every such cell needs its rungs walked
  (2) HISTORICAL SUPPORT: today's literal state is XLI r5 rank 2.4 with BOTH
      XLB and XLE at 52-week highs. How many times has that happened, and is
      today inside the support of the cell being priced?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)

px = close_panel(["XLI", "XLB", "XLE", "SPY"]).dropna(how="any")
idx = px.index
r5 = pct_rank(px["XLI"], 5, 252)
offhi = {t: px[t] / rolling_on_valid(px[t], lambda x: x.rolling(252).max()) - 1.0
         for t in ("XLB", "XLE")}
ANY = (offhi["XLB"] >= -0.0025) | (offhi["XLE"] >= -0.0025)


def stat(mask, legs, h, min_gap=None):
    ret = vehicle_ret(px, legs, h, 1)
    sig = idx[np.asarray(mask.reindex(idx, fill_value=False).values, bool)
              & ret.notna().values]
    if len(sig) == 0:
        return {"n_days": 0, "n": 0}
    e = declusters(sig, min_gap or max(h, 5), idx)
    v = ret.loc[e].values
    r = summarize(v, "")
    r["n_days"] = len(sig)
    r["ctrl_b"] = round(100 * ret.dropna().mean(), 3)
    r["edge"] = round(r["mean_pct"] - r["ctrl_b"], 3)
    r.pop("label")
    return r


print("=" * 100)
print("1a. XLI r5-RANK RUNG ladder, gate ON, h=5 (today's XLI r5 rank is 2.4)")
print("=" * 100)
rows = []
for rung in (1, 2, 3, 5, 7, 10, 15, 20, 30):
    for legs, lab in [([("XLI", 1.0)], "outright"), ([("XLI", 1.0), ("XLB", -1.0)], "pair-XLB")]:
        s = stat((r5 <= rung) & ANY, legs, 5)
        s["rung"] = rung; s["form"] = lab
        rows.append(s)
df = pd.DataFrame(rows)[["rung", "form", "n_days", "n", "mean_pct", "hit", "t", "edge", "worst_pct"]]
print(df.to_string(index=False))
print("\n  TODAY'S RANK IS 2.4 -> it lands in the <=3 rung. Read that row.")

print("\n" + "=" * 100)
print("1b. PEER-PROXIMITY rung, gate ON at r5rank<=5, h=5")
print("=" * 100)
rows = []
for prox in (0.0000, 0.0025, 0.0050, 0.0100, 0.0200, 0.0300):
    m = (r5 <= 5) & ((offhi["XLB"] >= -prox) | (offhi["XLE"] >= -prox))
    s = stat(m, [("XLI", 1.0)], 5); s["prox_pct"] = 100 * prox
    rows.append(s)
print(pd.DataFrame(rows)[["prox_pct", "n_days", "n", "mean_pct", "hit", "t", "edge"]]
      .to_string(index=False))

print("\n" + "=" * 100)
print("1c. r5 LOOKBACK ladder (the washout window itself), gate ON, h=5")
print("=" * 100)
rows = []
for k in (3, 4, 5, 6, 8, 10):
    rk = pct_rank(px["XLI"], k, 252)
    s = stat((rk <= 5) & ANY, [("XLI", 1.0)], 5)
    s["k"] = k; s["live_rank"] = round(float(rk.iloc[-1]), 1)
    rows.append(s)
print(pd.DataFrame(rows)[["k", "live_rank", "n_days", "n", "mean_pct", "hit", "t", "edge"]]
      .to_string(index=False))

print("\n" + "=" * 100)
print("2. HISTORICAL SUPPORT for today's LITERAL state")
print("=" * 100)
BOTH = (r5 <= 5) & (offhi["XLB"] >= -0.0025) & (offhi["XLE"] >= -0.0025)
print("  XLI r5rank<=5 AND XLB at a 52wh AND XLE at a 52wh: %d days ever" % int(BOTH.sum()))
print("  dates:", ", ".join(str(d.date()) for d in idx[BOTH.values]))
LOOSE = (r5 <= 10) & (offhi["XLB"] >= -0.01) & (offhi["XLE"] >= -0.01)
print("  loosened (rank<=10, peers within 1%%): %d days" % int(LOOSE.sum()))
print("  dates:", ", ".join(str(d.date()) for d in idx[LOOSE.values][-25:]))
for legs, lab in [([("XLI", 1.0)], "long XLI"), ([("XLI", 1.0), ("XLB", -1.0)], "PAIR   ")]:
    s = stat(LOOSE, legs, 5)
    if s.get("n", 0) == 0:
        print("  loosened cell, %s h=5: N=0 with a TRADEABLE forward return -- the only"
              " day in the cell is today, so there is NO history to price." % lab)
    else:
        print("  loosened cell, %s h=5: N_epi=%d mean %+.3f%% hit %.0f%% edge %+.3f pp"
              % (lab, s["n"], s["mean_pct"], s["hit"], s["edge"]))

print("\n" + "=" * 100)
print("3. THE REGISTRY QUESTION, finished: XLI washout by ERA against the family")
print("=" * 100)
for h in (3, 5):
    ret = fwd_lag(px["XLI"], h, 1)
    sig = idx[(r5 <= 5).values & ret.notna().values]
    e = declusters(sig, h, idx)
    v = ret.loc[e].values
    show(era_split(e, v), f"3.{h} plain XLI r5rank<=5 long, h={h}")
    print("   all-days control h=%d: %+.3f%%" % (h, 100 * ret.dropna().mean()))
    d = pd.DatetimeIndex(e)
    for lo, hi in [(2000, 2008), (2008, 2013), (2013, 2018), (2018, 2022), (2022, 2027)]:
        m = (d.year >= lo) & (d.year < hi)
        r = summarize(v[m], f"   {lo}-{hi-1}")
        print("   %s N=%3d mean %+.3f%% hit %.0f%%"
              % (r["label"], r["n"], r.get("mean_pct", float("nan")), r.get("hit", float("nan"))))
