"""Idea candidate for Thursday 2026-09-03: long USDJPY (short yen) after a
broad 2-ATR yen rally.

Tonight's brief: EURJPY, GBPJPY, CHFJPY, NZDJPY each closed >2 ATR lower on
2026-09-02. Four-or-more at once = 16 declustered sessions since 2005; USDJPY
higher the next session on 14, +0.538%, lag 0 from the cluster close.

Tradeable forms measured under the posts/pitch convention:
  A. brief reproduction, close(D) -> close(D+1), lag 0
  B. MOO D+1 -> MOC D+1 (the yfinance FX bar opens ~17:00 ET, so the open of
     D+1 sits next to the close of D; the gap it forfeits is printed)
  C. lag-1: close(D+1) -> close(D+2..D+5) (the true pitch convention, in case
     tomorrow is too late)
  D. h=5 from each anchor
All against the all-days base and a 126-session local control, era split,
concentration, worst, plus the 3-of-6 looser state for a bigger sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    cluster_note, declusters, era_split, fwd_lag, load_prices, local_control,
    sign_test, summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-02")
CROSSES = ["EURJPY=X", "GBPJPY=X", "CHFJPY=X", "NZDJPY=X", "AUDJPY=X", "JPY=X"]
px = load_prices(CROSSES)
usd = px["JPY=X"]
c = usd["Close"].dropna()
o = usd["Open"].reindex(c.index)
atr = pd.Series(wilder_atr(usd["High"], usd["Low"], usd["Close"]), index=usd.index).reindex(c.index)
print(f"tonight USDJPY {c.iloc[-1]:.3f} bar {c.index[-1].date()}  Wilder-14 ATR {atr.iloc[-1]:.4f} "
      f"({100*atr.iloc[-1]/c.iloc[-1]:.2f}%)  1d {100*(c.iloc[-1]/c.iloc[-2]-1):+.2f}%")

flags = {}
for t in CROSSES:
    d = px[t]
    a = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"]), index=d.index).shift(1)
    move = d["Close"] - d["Close"].shift(1)
    flags[t] = (move <= -2.0 * a)
F = pd.DataFrame(flags).fillna(False).reindex(c.index).fillna(False)
count = F.sum(axis=1)
print(f"tonight's 2-ATR-down crosses: {int(count.iloc[-1])} -> {[t for t in CROSSES if F[t].iloc[-1]]}")


def block(name, r, s, h, lag):
    r = r.dropna()
    if len(r) == 0:
        print(f"  {name:<46} n=0")
        return r
    st = summarize(r.values)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<46} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(r)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(r)):.4f}  "
          f"| all {100*allr.mean():+.3f}% hit {100*(allr>0).mean():.1f}%  local {100*loc.mean():+.3f}% "
          f"hit {100*(loc>0).mean():.1f}%  | worst {st['worst_pct']:+.2f}% ({r.idxmin().date()})")
    return r


def splits(r):
    r = r.dropna()
    v = r.values
    print("    era:", [(e["label"], e["n"], round(e.get("mean_pct", np.nan), 3),
                        round(e.get("hit", np.nan), 1)) for e in era_split(r.index, v)])
    print("    concentration:", cluster_note(r.index, v))
    print("    dates:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in r.items()])


p = pd.Series(np.arange(len(c)), index=c.index)
for thr in (4, 3):
    trig = c.index[(count >= thr).values]
    trig = trig[trig < ASOF]
    epi = declusters(trig, 5, c.index)
    print(f"\n########## {thr}+ crosses 2-ATR down: {len(trig)} days, {len(epi)} episodes ##########")
    print("=== A. brief reproduction: lag-0 h1 close-to-close ===")
    r = block("USDJPY lag0 h1", fwd_lag(c, 1, 0).reindex(epi), c, 1, 0)
    if thr == 4:
        splits(r)
    print("=== B. tradeable: MOO D+1 -> MOC D+1, and the gap forfeited ===")
    oc = pd.Series({d: c.iloc[p[d] + 1] / o.iloc[p[d] + 1] - 1 for d in epi if p[d] + 1 < len(c)})
    gap = pd.Series({d: o.iloc[p[d] + 1] / c.iloc[p[d]] - 1 for d in epi if p[d] + 1 < len(c)})
    st = summarize(oc.values)
    nup = int((oc > 0).sum())
    alloc = (c / o - 1).dropna()
    print(f"  open->close D+1                               n={st['n']}  mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
          f"{nup}-{len(oc)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(oc)):.4f}  "
          f"| all open->close {100*alloc.mean():+.3f}% hit {100*(alloc>0).mean():.1f}%  | worst {st['worst_pct']:+.2f}%")
    print(f"  overnight gap into D+1 (forfeited)             mean={100*gap.mean():+.3f}%  up {int((gap>0).sum())}-{int((gap<=0).sum())}  "
          f"max |gap| {100*gap.abs().max():.3f}%")
    print("=== C. lag-1 (close D+1 entry), h=1..4 ===")
    for h in (1, 2, 3, 4):
        block(f"USDJPY lag1 h{h}", fwd_lag(c, h, 1).reindex(epi), c, h, 1)
    print("=== D. lag-0 h5 ===")
    block("USDJPY lag0 h5", fwd_lag(c, 5, 0).reindex(epi), c, 5, 0)
    if thr == 4:
        print("=== E. the crosses themselves, lag-0 h1 (does the yen give it back everywhere?) ===")
        for t in ["EURJPY=X", "AUDJPY=X"]:
            cc = px[t]["Close"].dropna()
            block(f"{t} lag0 h1", fwd_lag(cc, 1, 0).reindex(epi), cc, 1, 0)

# placebo: a single-cross 2-ATR down day (EURJPY only), not part of a 4+ cluster
print("\n=== F. placebo: EURJPY alone 2-ATR down, cluster count <= 2, declustered 5 ===")
solo = c.index[(F["EURJPY=X"] & (count <= 2)).values]
solo = declusters(solo[solo < ASOF], 5, c.index)
block("USDJPY lag0 h1, solo EURJPY 2-ATR", fwd_lag(c, 1, 0).reindex(solo), c, 1, 0)
