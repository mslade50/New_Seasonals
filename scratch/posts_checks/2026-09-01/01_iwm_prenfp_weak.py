"""Idea candidates for Wednesday 2026-09-02, small caps into payrolls.

Tonight's context brief: IWM closed with a 5-day return rank of 6.7 (down
2.89% vs SPY -0.54%). On the 28 pre-payrolls anchors (3 td out) with IWM's
rank in the bottom decile, IWM rose 20 of 28 the NEXT session (+0.50%,
lag 0), then handed it back: 10-18 by the third session, 7-21 against SPY
by the fifth at -0.819%.

Two tradeable forms are measured here under the pitch convention:
  LONG  : the brief's h1 is tomorrow's own session, so the only tradeable
          version is MOO Wed -> MOC Wed (open-to-close), which forfeits the
          overnight gap. Printed with the gap component beside it.
  SHORT : MOC Wed entry (lag-1), out h=1..4 sessions later, absolute IWM
          and IWM minus SPY. h=2 is the print session, h=4 is the session
          after Labor Day.
Both against the unconditioned pre-NFP parent, all-days and local controls,
eras, midterm cut, concentration, worst.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    anchor_positions, cluster_note, era_split, fwd_lag, load_events,
    load_prices, local_control, pct_rank, sign_test, summarize, wilder_atr,
)

ASOF = pd.Timestamp("2026-09-01")
px = load_prices(["IWM", "SPY"])
iwm, spy = px["IWM"], px["SPY"]
c = iwm["Close"].dropna()
o = iwm["Open"].reindex(c.index)
sc = spy["Close"].reindex(c.index)
atr = pd.Series(wilder_atr(iwm["High"], iwm["Low"], iwm["Close"]), index=iwm.index).reindex(c.index)
r5rank = pct_rank(c, 5, 252)
print(f"tonight IWM {c.iloc[-1]:.2f} bar {c.index[-1].date()}  rank5 {r5rank.iloc[-1]:.1f}  "
      f"Wilder-14 ATR {atr.iloc[-1]:.4f} ({100*atr.iloc[-1]/c.iloc[-1]:.2f}%)")

nfp = load_events(["nfp"])["date"]
nfp = nfp[nfp <= pd.Timestamp("2026-12-31")]
pos3, kept = anchor_positions(c.index, nfp, offset=-3)
anch = c.index[pos3]
anch = anch[anch < ASOF]
weak = anch[(r5rank.reindex(anch) <= 10).values]
print(f"pre-NFP anchors (3 td out): {len(anch)}   with IWM rank5 <= 10: {len(weak)}  "
      f"{weak[0].date()}..{weak[-1].date()}")
nxt = c.index[c.index > ASOF] if (c.index > ASOF).any() else None
print(f"tonight is an anchor: {ASOF in set(anch) or (c.index[pos3[-1]] == ASOF if pos3 else False)}")


def block(name, r, s, h, lag=1):
    r = r.dropna()
    if len(r) == 0:
        print(f"  {name:<44} n=0")
        return r
    st = summarize(r.values)
    nup = int((r > 0).sum())
    allr = fwd_lag(s, h, lag).dropna()
    loc = allr.reindex(local_control(s.index, r.index, 126)).dropna()
    print(f"  {name:<44} n={st['n']:<4} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
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
    mid = r[[d.year % 4 == 2 for d in r.index]]
    nu = int((mid > 0).sum())
    if len(mid):
        print(f"    midterm n={len(mid)} mean={100*mid.mean():+.3f}% {nu}-{len(mid)-nu} sp={sign_test(nu, len(mid)):.4f}")


print("\n=== A. brief reproduction: lag-0 h1 close-to-close from the anchor (expect 20-28, +0.50%) ===")
block("IWM lag0 h1, weak-decile pre-NFP", fwd_lag(c, 1, 0).reindex(weak), c, 1, 0)
block("IWM lag0 h1, ALL pre-NFP anchors", fwd_lag(c, 1, 0).reindex(anch), c, 1, 0)

print("\n=== B. LONG tradeable form: MOO t+1 -> MOC t+1, with the overnight gap it forfeits ===")
p = pd.Series(np.arange(len(c)), index=c.index)
oc = pd.Series({d: c.iloc[p[d] + 1] / o.iloc[p[d] + 1] - 1 for d in weak if p[d] + 1 < len(c)})
gap = pd.Series({d: o.iloc[p[d] + 1] / c.iloc[p[d]] - 1 for d in weak if p[d] + 1 < len(c)})
st = summarize(oc.values)
nup = int((oc > 0).sum())
alloc = (c / o - 1).dropna()
print(f"  open->close t+1                              n={st['n']}  mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%  "
      f"{nup}-{len(oc)-nup} ({st['hit']:.1f}%)  t={st['t']:+.2f}  sp={sign_test(nup, len(oc)):.4f}  "
      f"| all open->close {100*alloc.mean():+.3f}% hit {100*(alloc>0).mean():.1f}%  | worst {st['worst_pct']:+.2f}%")
print(f"  overnight gap into t+1 (forfeited)            mean={100*gap.mean():+.3f}%  up {int((gap>0).sum())}-{int((gap<=0).sum())}")
splits(oc)
print("  named losers:", [(d.date().isoformat(), round(100 * x, 2)) for d, x in oc.sort_values().head(6).items()])

print("\n=== C. SHORT form: MOC t+1 entry (lag-1), IWM absolute, h=1..4 (h2 = the print) ===")
for h in (1, 2, 3, 4, 5):
    block(f"IWM lag1 h{h}, weak-decile pre-NFP", fwd_lag(c, h, 1).reindex(weak), c, h)
r3 = fwd_lag(c, 3, 1).reindex(weak)
print("  -- h3 splits --")
splits(r3)
r4 = fwd_lag(c, 4, 1).reindex(weak)
print("  -- h4 splits --")
splits(r4)
print("  h4 named winners for a short (most negative):",
      [(d.date().isoformat(), round(100 * x, 2)) for d, x in r4.dropna().sort_values().head(6).items()])
print("  h4 named losers for a short (most positive):",
      [(d.date().isoformat(), round(100 * x, 2)) for d, x in r4.dropna().sort_values().tail(6).items()])

print("\n=== C2. the same short as IWM minus SPY (lag-1) ===")
for h in (2, 3, 4, 5):
    rel = (fwd_lag(c, h, 1) - fwd_lag(sc, h, 1)).reindex(weak).dropna()
    nup = int((rel > 0).sum())
    st = summarize(rel.values)
    print(f"  IWM-SPY lag1 h{h}                              n={st['n']}  mean={st['mean_pct']:+.3f}%  "
          f"{nup}-{len(rel)-nup}  t={st['t']:+.2f}  sp={sign_test(len(rel)-nup, len(rel)):.4f} (short side)")

print("\n=== D. the parent: ALL pre-NFP anchors, lag-1, h=1..4 ===")
for h in (1, 2, 3, 4):
    block(f"IWM lag1 h{h}, ALL pre-NFP", fwd_lag(c, h, 1).reindex(anch), c, h)

print("\n=== E. placebo: weak-decile IWM on NON pre-NFP days (rank<=10 anywhere, min-gap 5), lag-1 h3/h4 ===")
from pitch_lab import declusters  # noqa: E402
anyweak = c.index[(r5rank <= 10).fillna(False).values]
anyweak = declusters(anyweak.difference(anch), 5, c.index)
anyweak = anyweak[anyweak < ASOF]
for h in (1, 3, 4):
    block(f"IWM lag1 h{h}, weak-decile, not pre-NFP", fwd_lag(c, h, 1).reindex(anyweak), c, h)
