"""C3 round 2 - mandatory pass on the pitched XLU cell after round 1 said the
rates gate is worth +0.086pp at h=3 and the LIVE (SPY-near-high) slice is
negative. This checks whether ANY reasonable neighbour rescues it:
  (a) threshold + lookback nudges on both legs
  (b) drop-top-k / concentration / best-year share
  (c) midterm and fragility-dial splits
  (d) the SPY-near-high threshold walk (today -1.85% off the high)
  (e) the inverse cell ("the bond WAS hit"), stated for the record as a
      post-hoc sign flip recovered from a kill report -- registry rule
      2026-08-07 -- and NOT live today anyway (TLT rank21 = 48.4).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

px = close_panel(["XLU", "SPY", "TLT", "IEF"])
px = px[px.index >= "2002-07-30"]
idx = px.index
LEGS = [("XLU", 1.0)]

rk = {t: pct_rank(px[t], 21) for t in ["XLU", "TLT", "SPY"]}
r21 = {t: _valid_pct_change(px[t], 21) for t in ["XLU", "TLT", "SPY"]}
hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
off_hi = px["SPY"] / hi - 1.0
DIAL = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = DIAL["63d"].rolling(10).mean()


def cellstats(mask, h, lbl, min_gap=21):
    ret = vehicle_ret(px, LEGS, h, 1)
    d = idx[mask.reindex(idx, fill_value=False).values].intersection(ret.dropna().index)
    if len(d) == 0:
        return {"label": lbl, "n": 0}, np.array([]), d
    ep = declusters(d, min_gap, idx)
    v = ret.loc[ep].values
    r = summarize(v, lbl)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 3)
    return r, v, ep


print("=== (a) NEIGHBOUR GRID: XLU washout threshold x TLT band, h=3 and h=5 ===")
for h in (3, 5):
    ret = vehicle_ret(px, LEGS, h, 1)
    base = 100 * ret.dropna().mean()
    rows = []
    for wt in (2, 5, 10, 15):
        for lo, hiq, tlab in [(0, 100, "no rates gate"), (25, 75, "TLT 25-75 PITCHED"),
                              (20, 80, "TLT 20-80"), (35, 65, "TLT 35-65"),
                              (30, 70, "TLT 30-70")]:
            m = (rk["XLU"] <= wt) & (rk["TLT"] >= lo) & (rk["TLT"] <= hiq)
            r, v, ep = cellstats(m, h, f"XLUrk<={wt:<2d} x {tlab}")
            if r.get("n"):
                r["edge_pp"] = round(r["mean_pct"] - base, 3)
            rows.append(r)
    show(rows, f"h={h}  all-days base {base:+.3f}%")
    got = [r for r in rows if r.get("n")]
    print(f"  -> {len(got)} cells scanned, mean edge "
          f"{np.mean([r['edge_pp'] for r in got]):+.3f}pp, "
          f"best {max(r['edge_pp'] for r in got):+.3f}pp, "
          f"worst {min(r['edge_pp'] for r in got):+.3f}pp")
    # what the rates gate is worth AT EVERY washout threshold
    for wt in (2, 5, 10, 15):
        a = [r for r in rows if r["label"].startswith(f"XLUrk<={wt:<2d}") and "no rates" in r["label"]][0]
        b = [r for r in rows if r["label"].startswith(f"XLUrk<={wt:<2d}") and "PITCHED" in r["label"]][0]
        if a.get("n") and b.get("n"):
            print(f"     rates gate at XLUrk<={wt}: {b['mean_pct']-a['mean_pct']:+.3f}pp "
                  f"(N {a['n']} -> {b['n']})")

print("\n=== (b) CONCENTRATION on the pitched cell ===")
m = (rk["XLU"] <= 5) & (rk["TLT"] >= 25) & (rk["TLT"] <= 75)
for h in (3, 5, 10):
    r, v, ep = cellstats(m, h, "pitched")
    o = np.argsort(-v)
    yr = pd.Series(v, index=ep).groupby(ep.year).sum()
    print(f"  h={h} N={len(v)} full {100*v.mean():+.3f}%  drop-top1 "
          f"{100*np.delete(v,o[:1]).mean():+.3f}%  drop-top2 "
          f"{100*np.delete(v,o[:2]).mean():+.3f}%  median {100*np.median(v):+.3f}%  "
          f"record {int((v>0).sum())}-{int((v<=0).sum())}")
    print(f"       {cluster_note(ep, v)}")

print("\n=== (c) MIDTERM + FRAGILITY DIAL (recompute vintage pre-2026-07-02) ===")
for h in (3, 5):
    r, v, ep = cellstats(m, h, "pitched")
    mid = np.array([(x.year % 4) == 2 for x in ep])
    show([summarize(v[mid], f"h={h} midterm (N={int(mid.sum())})  <-- 2026"),
          summarize(v[~mid], f"h={h} non-midterm (N={int((~mid).sum())})")], "")
    dv = ma10.reindex(ep)
    have = dv.notna().values
    if have.sum():
        hi_ = have & (dv.values >= 50)
        lo_ = have & (dv.values < 50)
        show([summarize(v[hi_], f"h={h} dial>=50 (N={int(hi_.sum())})"),
              summarize(v[lo_], f"h={h} dial<50 (N={int(lo_.sum())})")], "")
        print(f"   episodes with a dial: {int(have.sum())} of {len(ep)}; "
              f"above 70: {int((dv.dropna()>=70).sum())}; TODAY 89.5")
        print("   " + ", ".join(f"{str(x.date())}={dv[x]:.0f}" for x in ep[have]))

print("\n=== (d) SPY-NEAR-HIGH WALK (today -1.85% off the 52w high) ===")
for h in (3, 5):
    rows = []
    for thr in (-0.01, -0.02, -0.03, -0.05, -0.10):
        mm = m & (off_hi >= thr)
        r, v, ep = cellstats(mm, h, f"pitched x SPY >= {100*thr:.0f}% off hi")
        rows.append(r)
    mm = m & (off_hi < -0.03)
    r, v, ep = cellstats(mm, h, "pitched x SPY < -3% off hi (complement)")
    rows.append(r)
    show(rows, f"h={h}")

print("\n=== (e) THE INVERSE CELL, for the record only ===")
inv = (rk["XLU"] <= 5) & (rk["TLT"] < 25)
for h in (3, 5, 10):
    r, v, ep = cellstats(inv, h, f"XLU washout x TLT ALSO hit, h={h}")
    print(f"  {r['label']}: N={r['n']} mean {r['mean_pct']:+.3f}% hit {r['hit']:.1f}% "
          f"sign p {r['sign_p']}  | LIVE TODAY? TLT rank21 = {rk['TLT'].iloc[-1]:.1f} -> "
          f"{'YES' if rk['TLT'].iloc[-1] < 25 else 'NO'}")
print("  This is the negation of the pitched mechanism and is NOT live. Taking it "
      "would be a post-hoc sign flip recovered from a kill report (registry "
      "2026-08-07) on top of an eighth dead utilities expression.")
