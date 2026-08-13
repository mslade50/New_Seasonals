"""C1 round 2b - the ONE cell today actually sits in.

Today's spread is +27.4% AND its 252d pctile is 98.4, so today is in the
INTERSECTION of the two definitions, and that intersection was the best cell
in a1b (post-cut h=5 +1.561% N=10, h=10 +2.096% N=10 at a 90% hit,
bootstrap 0.019). Killing C1 on "the abs-level definition is negative" would
be dishonest if the negative came from episodes today is not in. So:

 1. list the episodes of each cell with their returns, and name the -14.63%
    one that flips the abs>=25% mean.
 2. run the placebo anchor ladder ON THE INTERSECTION CELL. If the real anchor
    is dominated there too, the kill is the anchor, not the definition.
 3. sleeve overlap + month composition of the intersection episodes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (  # noqa: E402
    bootstrap_p_le0, cluster_note, declusters, load_prices, show, sign_test,
    summarize, fwd_lag,
)

LEV_CUT = pd.Timestamp("2018-03-01")
px = load_prices(["^VIX", "^VIX3M", "SVXY", "SPY"])
vx, v3 = px["^VIX"]["Close"].dropna(), px["^VIX3M"]["Close"].dropna()
svxy, spy = px["SVXY"]["Close"].dropna(), px["SPY"]["Close"].dropna()
spread = (v3 / vx - 1.0).dropna()
pct252 = spread.rolling(253).apply(lambda w: (w.iloc[-1] > w[:-1]).mean() * 100.0,
                                   raw=False)
pv = pd.DataFrame({"SVXY": svxy, "SPY": spy}).dropna()
sub = pv[pv.index >= LEV_CUT]
idx = sub.index
pos = pd.Series(range(len(idx)), index=idx)

cells = {
    "pctile>=98 only": (pct252 >= 98) & (spread < 0.25),
    "abs>=25% only": (spread >= 0.25) & (pct252 < 98),
    "INTERSECTION (today)": (pct252 >= 98) & (spread >= 0.25),
}
print("=== 1. episode-by-episode, post-cut, long SVXY ===")
for h in (5, 10):
    ret = fwd_lag(sub["SVXY"], h, 1)
    ok = ret.notna()
    for lbl, m in cells.items():
        t = idx[m.reindex(idx, fill_value=False).values & ok.values]
        if len(t) == 0:
            print(f"  h={h} {lbl}: NO post-cut days")
            continue
        epi = declusters(t, 21, idx[ok.values])
        v = ret.loc[epi].values
        print(f"\n  h={h} {lbl}: N={len(epi)} mean {100*v.mean():+.3f}% "
              f"median {100*np.median(v):+.3f}% hit {100*(v>0).mean():.0f}% "
              f"worst {100*v.min():+.2f}% boot {bootstrap_p_le0(v):.3f}")
        for d, x in zip(epi, v):
            print(f"      {d.date()}  {100*x:+7.2f}%   spread "
                  f"{100*spread.get(d, np.nan):5.1f}%  pctile "
                  f"{pct252.get(d, np.nan):5.1f}")

print("\n=== 2. placebo ladder ON THE INTERSECTION CELL (post-cut) ===")
mI = cells["INTERSECTION (today)"]
for h in (5, 10):
    ret = fwd_lag(sub["SVXY"], h, 1)
    ok = ret.notna()
    base_t = idx[mI.reindex(idx, fill_value=False).values]
    lad = []
    for k in range(-10, 11):
        sh = []
        for d in base_t:
            p = pos.get(d)
            if p is None:
                continue
            q = p + k
            if 0 <= q < len(idx) and ok.iloc[q]:
                sh.append(idx[q])
        s = pd.DatetimeIndex(sorted(set(sh)))
        epi = declusters(s, 21, idx[ok.values])
        v = ret.loc[epi].values
        lad.append({"k": k, "n": len(epi), "mean_pct": round(100 * v.mean(), 3),
                    "hit": round(100 * float((v > 0).mean()), 1),
                    "worst": round(100 * v.min(), 2)})
    df = pd.DataFrame(lad)
    real = df.loc[df.k == 0, "mean_pct"].iloc[0]
    rank = int((df["mean_pct"] >= real).sum())
    print(f"\n  h={h}: real k=0 = {real:+.3f}%  rank {rank}/{len(df)} "
          f"(empirical p {rank/len(df):.3f})")
    print(df.to_string(index=False))

print("\n=== 3. intersection episodes: month + sleeve overlap + SPY residual ===")
ev = pd.read_csv(Path(__file__).resolve().parents[3] / "data/macro_events.csv",
                 parse_dates=["date"])
opex = ev[ev["event"] == "opex"]["date"]
for h in (5, 10):
    rs = fwd_lag(sub["SVXY"], h, 1)
    rm = fwd_lag(sub["SPY"], h, 1)
    ok = rs.notna() & rm.notna()
    t = idx[mI.reindex(idx, fill_value=False).values & ok.values]
    epi = declusters(t, 21, idx[ok.values])
    ctl = idx[ok.values].difference(t)
    b = np.polyfit(rm.loc[ctl].values, rs.loc[ctl].values, 1)
    resid = rs - (b[1] + b[0] * rm)
    v = resid.loc[epi].values
    base = float((resid.loc[ctl] > 0).mean())
    w = int((v > 0).sum())
    mon = pd.DatetimeIndex(epi).month
    nd = np.isin(mon, [11, 12])
    v4 = np.array([bool((((opex - d).dt.days >= -1) & ((opex - d).dt.days <= 5)).any())
                   and d.month != 9 for d in epi])
    own = nd | v4
    print(f"\n  h={h}: N={len(epi)}  beta {b[0]:.2f}  resid {100*v.mean():+.3f}% "
          f"hit {100*(v>0).mean():.0f}% (base {100*base:.1f}%) "
          f"sign p vs base {sign_test(w, len(v), base):.4f}")
    print(f"    months {sorted(pd.Series(mon).value_counts().to_dict().items())};"
          f" Nov/Dec {int(nd.sum())}; sleeve-owned (V2 or V4) {int(own.sum())}/{len(epi)}")
    raw = rs.loc[epi].values
    print(f"    raw {100*raw.mean():+.3f}%   ex-sleeve raw "
          f"{100*raw[~own].mean() if (~own).sum() else float('nan'):+.3f}% "
          f"(N={int((~own).sum())})")
    print(f"    {cluster_note(epi, raw)}")
    show([summarize(raw, "intersection raw"),
          summarize(rs[ok].values, "CTRL all post-cut days")])

print("\n=== 4. today's own numbers under the winning cell ===")
print(f"  spread {100*spread.iloc[-1]:.1f}%  pctile252 {pct252.iloc[-1]:.1f}  "
      f"VIX {vx.iloc[-1]:.2f}  SVXY trailing-21d "
      f"{100*svxy.pct_change(21).iloc[-1]:+.2f}%")
print("  episode median trailing-21d SVXY on intersection days:")
tI = idx[mI.reindex(idx, fill_value=False).values]
print(f"    {100*svxy.pct_change(21).reindex(tI).median():+.2f}%  "
      f"(today {100*svxy.pct_change(21).iloc[-1]:+.2f}%)")
