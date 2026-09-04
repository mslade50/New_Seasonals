"""Redo the two surviving cells on each ticker's OWN session index.

`close_panel` builds a UNION index, so a 252-bar rolling max for SPY includes
rows where SPY did not trade (holidays other markets kept). That moved the
at-a-52w-high mask between drills 07 (n=76) and 08 (n=90) purely because 08
added ^RUT to the panel. Neither number is quotable. This recomputes both
findings from `load_prices`, one ticker at a time, on native NYSE sessions,
and those are the numbers that go in the brief.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    load_prices, load_events, fwd_ret, summarize, era_split, sign_test,
    cluster_note,
)

px = load_prices(["SPY", "^VIX", "TLT", "^TNX", "CL=F", "IEF"])
close = {t: px[t]["Close"].dropna().sort_index() for t in px}
for t, s in close.items():
    print(f"  {t:8s} {len(s):5d} bars {s.index.min().date()} -> {s.index.max().date()}")

spy = close["SPY"]
spy_dh = spy / spy.rolling(252).max() - 1.0
print(f"\nSPY dist-52w-high today: {100*spy_dh.iloc[-1]:+.4f}%  "
      f"(sessions at > -0.5%: {int((spy_dh > -0.005).sum())})")
cl = close["CL=F"]
cl21 = cl / cl.shift(21) - 1.0
print(f"CL=F 21d today: {100*cl21.iloc[-1]:+.2f}%")

ev = load_events(["cpi"])


def anchors_k(sess: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    """The session k td before each CPI, on THIS instrument's own calendar."""
    out = []
    for d in ev["date"]:
        d = pd.Timestamp(d)
        pos = sess.searchsorted(d)
        if pos >= len(sess) or sess[pos] != d or pos - k < 0:
            continue
        out.append(sess[pos - k])
    return pd.DatetimeIndex(out)


def stat(s, idx, h, label, indent="   ", extra=False):
    f = fwd_ret(s, h)
    v = f.reindex(pd.DatetimeIndex(sorted(set(idx)))).dropna()
    if len(v) < 3:
        print(f"{indent}{label:56s} n={len(v)} too few")
        return None
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"{indent}{label:56s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
          f"dn-p {sign_test(len(v)-up, len(v)):.4f}")
    if extra:
        print(f"        {cluster_note(v.index, v.values)}")
        for e in era_split(v.index, v.values):
            if e.get("n", 0):
                print(f"        era n={e['n']:4d} mean {e['mean_pct']:+.3f}% "
                      f"hit {e['hit']:.1f}% t={e['t']:+.2f}")
        keep = v.iloc[np.argsort(-np.abs(v.values))[2:]]
        rk = summarize(keep.values, "")
        upk = int((keep.values > 0).sum())
        print(f"        ex the 2 largest: n={rk['n']} mean {rk['mean_pct']:+.3f}% "
              f"hit {rk['hit']:.1f}% {upk}-{len(keep)-upk}")
        yrs = pd.Series(v.values > 0).groupby(v.index.year).mean()
        print(f"        hit by era-decade: "
              f"{ {d: round(100*yrs[(yrs.index//10)*10 == d].mean(), 0) for d in sorted(set((yrs.index//10)*10))} }")
    return v


print("\n" + "=" * 78)
print("FINDING 1. VIX on the CPI eve when SPY closed at a 52w high")
print("  anchor = the session 2 td before a CPI (today's analogue); h1 = the eve")
print("=" * 78)
sess = spy.index
a_all = anchors_k(sess, 2)
print(f"   CPI anchors on SPY sessions: {len(a_all)}")
hi = spy_dh.reindex(a_all).fillna(-1) > -0.005
a_hi = a_all[hi.values]
vix = close["^VIX"]
v = stat(vix, a_hi, 1, "^VIX, CPI eve with SPY at a 52w high  [THE CELL]", extra=True)
stat(spy, a_hi, 1, "   SPY over the same session")
print("\n   controls:")
stat(vix, sess, 1, "^VIX, all sessions")
stat(vix, a_all, 1, "^VIX, all CPI eves regardless of the tape")
stat(vix, a_all[~hi.values], 1, "^VIX, CPI eves with SPY NOT at a 52w high")
hi_all = sess[(spy_dh > -0.005).reindex(sess).fillna(False).values]
stat(vix, hi_all, 1, "^VIX, SPY at a 52w high on ANY session")
stat(vix, hi_all.difference(a_all), 1, "^VIX, SPY at a 52w high, NOT a CPI eve")
print("\n   how far off the high does the effect survive?")
for thr in (0.002, 0.005, 0.01, 0.02, 0.03):
    m = spy_dh.reindex(a_all).fillna(-1) > -thr
    stat(vix, a_all[m.values], 1, f"   SPY within {100*thr:.1f}% of its 52w high")

print("\n" + "=" * 78)
print("FINDING 2. TLT on the CPI session when crude ran hard into it")
print("  anchor = 2 td before a CPI; h2 = the CPI session's own close-to-close")
print("=" * 78)
tlt = close["TLT"]
sess_t = tlt.index
a_t = anchors_k(sess_t, 2)
print(f"   CPI anchors on TLT sessions: {len(a_t)}")
hot = cl21.reindex(a_t).fillna(-1) >= 0.10
v = stat(tlt, a_t[hot.values], 2, "TLT, CPI day, crude 21d >= +10%  [THE CELL]", extra=True)
stat(close["^TNX"], a_t[hot.values], 2, "   ^TNX over the same session")
stat(close["IEF"], a_t[hot.values], 2, "   IEF over the same session")
stat(tlt, a_t[hot.values], 1, "   TLT on the eve instead of the print")
print("\n   controls:")
stat(tlt, sess_t, 2, "TLT, all sessions, 2-day forward")
stat(tlt, a_t, 2, "TLT, all CPI days regardless of crude")
stat(tlt, a_t[~hot.values], 2, "TLT, CPI days with crude NOT hot")
hot_any = sess_t[(cl21 >= 0.10).reindex(sess_t).fillna(False).values]
stat(tlt, hot_any, 2, "TLT, crude 21d >= +10% on ANY session")
stat(tlt, hot_any.difference(a_t), 2, "TLT, crude hot, NOT a CPI eve")
print("\n   threshold walk on the crude condition:")
for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
    m = cl21.reindex(a_t).fillna(-1) >= thr
    stat(tlt, a_t[m.values], 2, f"   crude 21d >= +{100*thr:.0f}%")
print("\n   per-episode, the >= +15% cell (today is +15.25%):")
m15 = cl21.reindex(a_t).fillna(-1) >= 0.15
f2 = fwd_ret(tlt, 2).reindex(a_t[m15.values]).dropna()
for d, r in f2.items():
    print(f"      anchor {d.date()}  crude 21d {100*cl21.loc[d]:+6.1f}%  "
          f"TLT over the print {100*r:+.2f}%")
