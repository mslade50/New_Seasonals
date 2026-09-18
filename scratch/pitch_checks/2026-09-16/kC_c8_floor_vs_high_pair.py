"""kC c8 round 1: long the SPDRs at a 5/21/63 <= 10 triple floor (equal weight)
against short XLE, on days XLE closes at its trailing-252 high. lag=1, h=5/10,
declustered gap = h. Nine original SPDRs.

Decisive comparison demanded by the 2026-09-10 registry entry: the GENERIC
cross-sectional reversal pair (long the two worst 21d SPDRs, short the two
best, and long-two-worst / short-best-one) on the SAME dates. Paired diff.
Also the generalised label: any SPDR at a 252 high (short it) against the
floored members.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)
SP = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
px = close_panel(SP + ["SPY"]).dropna(subset=SP)
cal = px.index
spy200 = px["SPY"] / px["SPY"].rolling(200).mean() - 1.0

R = {t: (pct_rank(px[t], 5), pct_rank(px[t], 21), pct_rank(px[t], 63)) for t in SP}
FLOOR = pd.DataFrame({t: ((R[t][0] <= 10) & (R[t][1] <= 10) & (R[t][2] <= 10)) for t in SP}).fillna(False)
HI = pd.DataFrame({t: px[t] >= px[t].rolling(252).max() - 1e-12 for t in SP}).fillna(False)
RET21 = px[SP] / px[SP].shift(21) - 1.0


def fwd(h):
    return px[SP].shift(-(1 + h)) / px[SP].shift(-1) - 1.0


def run(h, min_floor=2, short_set="XLE"):
    F = fwd(h)
    valid = F.dropna().index
    if short_set == "XLE":
        m = (FLOOR.sum(axis=1) >= min_floor) & HI["XLE"]
    else:
        m = (FLOOR.sum(axis=1) >= min_floor) & (HI.sum(axis=1) >= 1)
    trig = cal[m.values].intersection(valid)
    epi = declusters(trig, h, valid)
    rows = []
    for d in epi:
        lo = [t for t in SP if FLOOR.loc[d, t]]
        hi = ["XLE"] if short_set == "XLE" else [t for t in SP if HI.loc[d, t]]
        lab = F.loc[d, lo].mean() - F.loc[d, hi].mean()
        r21 = RET21.loc[d].sort_values()
        w2, b2, b1 = list(r21.index[:2]), list(r21.index[-2:]), list(r21.index[-1:])
        g22 = F.loc[d, w2].mean() - F.loc[d, b2].mean()
        g21 = F.loc[d, w2].mean() - F.loc[d, b1].mean()
        rows.append({"date": d, "long": "+".join(lo), "short": "+".join(hi),
                     "labelled": lab, "long_leg": F.loc[d, lo].mean(), "short_leg": -F.loc[d, hi].mean(),
                     "gen_2v2": g22, "gen_2v1": g21, "worst2": "+".join(w2), "best1": b1[0],
                     "xle_is_best": b1[0] == "XLE", "above200": spy200.loc[d] > 0,
                     "midterm": d.year % 4 == 2})
    return pd.DataFrame(rows)


def s(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "N=0"
    w = int((v > 0).sum())
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
    return f"N={len(v):3d} mean {100*v.mean():+.3f}% hit {100*w/len(v):5.1f}% t {t:+.2f} sign p {sign_test(w, len(v)):.4f}"


print("live: floor members", [t for t in SP if FLOOR[t].iloc[-1]], " 252-high members", [t for t in SP if HI[t].iloc[-1]],
      " last", cal[-1].date())
# all-days control for the labelled vehicle's structure: XLU+XLI minus XLE on all days
for h in (5, 10):
    F = fwd(h).dropna()
    ctl = (F[["XLU", "XLI"]].mean(axis=1) - F["XLE"])
    print(f"\n######## h={h} ########  CTRL all-days (XLU+XLI)/2 - XLE: {100*ctl.mean():+.3f}%")
    for mf, ss in [(2, "XLE"), (1, "XLE"), (2, "ANY"), (1, "ANY")]:
        d = run(h, mf, ss)
        print(f"\n  -- >= {mf} floor members, short {'XLE at 252 high' if ss=='XLE' else 'any SPDR at 252 high'} --")
        if d.empty:
            print("    no episodes")
            continue
        print("   labelled pair     ", s(d.labelled))
        print("     long leg        ", s(d.long_leg), "| short leg", s(d.short_leg))
        print("   generic 2w-v-2b   ", s(d.gen_2v2))
        print("   generic 2w-v-1b   ", s(d.gen_2v1))
        diff = d.labelled - d.gen_2v2
        dt = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 2 else np.nan
        print(f"   paired labelled - generic2v2 {100*diff.mean():+.3f}% (t {dt:+.2f}); "
              f"XLE is the 21d leader on {int(d.xle_is_best.sum())}/{len(d)}" if ss == "XLE" else
              f"   paired labelled - generic2v2 {100*diff.mean():+.3f}% (t {dt:+.2f})")
        for lbl, m in [("above200", d.above200), ("below200", ~d.above200), ("midterm", d.midterm),
                       ("midterm&above200", d.midterm & d.above200), ("pre-2018", d.date < "2018"),
                       ("2018+", d.date >= "2018")]:
            print(f"     {lbl:18s} labelled {s(d.labelled[m])}")
        if mf == 2 and ss == "XLE" or (mf == 1 and ss == "XLE" and h == 10):
            print(d.assign(**{c: (100 * d[c]).round(2) for c in ["labelled", "long_leg", "short_leg", "gen_2v2", "gen_2v1"]})
                  [["date", "long", "short", "labelled", "long_leg", "short_leg", "gen_2v2", "best1", "above200"]].to_string(index=False))
